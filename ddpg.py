import tensorflow as tf
import numpy as np
import gym
import random

class OU(object):
	def __init__(self,action_space, theta=0.15, sigma=0.2, mu = 0):
		self.theta = theta
		self.sigma = sigma
		self.mu = mu
		self.dt = 1
		self.multt = self.dt * self.theta
		self.multm = self.dt * self.mu
		self.mults = np.sqrt(self.dt) * self.sigma
		self.k = action_space.shape[0]
		self.state = tf.Variable(np.zeros(self.k),trainable=False,dtype=tf.float32)
	def sample(self):
		inc = tf.assign_add(self.state, 
			self.multm - self.multt * self.state + tf.random.normal((self.k,),stddev=self.mults))
		return inc
	def reset(self):
		return tf.assign(self.state, np.zeros(self.k))

class Config(object):
	def __init__(self,action_dim,
				 rspace=1,
				 hidden=[100,80,50],
				 clip=False,
				 actLayer=1, 
				 init=3e-3,
				 gamma=0.90, 
				 tau=0.001, 
				 lr=1e-3,
				 l2=1e-4):
		self.action_dim = action_dim
		self.rspace = rspace
		self.hidden = hidden
		self.actLayer = actLayer
		self.init = tf.random_uniform_initializer(minval=-init,
												maxval=init)
		self.clip = clip
		self.gamma = gamma
		self.tau = tau
		self.lr = lr
		self.l2 = l2

class ReplayBuffer(object):
	def __init__(self,size):
		self.size = size
		self.buffer = []
		self.position = 0
	def push(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.size:
			self.buffer.append([state,action,reward,next_state,done])
			return
		self.buffer[self.position] = [state, action, reward, next_state, done]
		self.position = (self.position + 1) % self.size
	def get(self,k):
		batch = random.sample(self.buffer, k)
		state, action, reward, next_state, done = map(np.vstack, zip(*batch))
		return state, action, reward, next_state, done
	def __len__(self):
		return len(self.buffer)

class Base(object):
	def __init__(self, config, name):
		self.config = config
		self.name = name
		self.global_step = tf.Variable(0, name=self.name+'_global_step',trainable=False)
	def train(self, loss):
		t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
		optimizer = tf.contrib.opt.AdamWOptimizer(self.config.l2, learning_rate=self.config.lr)
		decay_vars = [v for v in t_vars if 'batch_normalization' not in v.name]
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
		with tf.control_dependencies(update_ops):
			gradients = tf.gradients(loss,t_vars)
			if self.config.clip:
				gradients, _ = tf.clip_by_global_norm(gradients, 1)
			train = optimizer.apply_gradients(zip(gradients,t_vars), 
										  global_step=self.global_step, 
										  decay_var_list=decay_vars)
		return train
	def copy_to_target(self,original,search=None):
		orig_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=original.name)
		target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		if search:
			copy_ops = [tf.assign(t,o) for o,t in zip(orig_vars,target_vars) if search in o.name]
		else:
			copy_ops = [tf.assign(t,o) for o,t in zip(orig_vars,target_vars)]
		return copy_ops
	def update_target(self,original):
		orig_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=original.name)
		target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
		update_ops = []
		for o,t in zip(orig_vars,target_vars):
			update_ops.append(tf.assign(t, self.config.tau * o + (1 - self.config.tau) * t))
		copy_ops = self.copy_to_target(original,'moving')
		update_ops.extend(copy_ops)
		return update_ops

class Critic(Base):
	def __init__(self,config,name):
		self.name = name
		super().__init__(config,name)
	def _build_network(self,x,is_training):
		with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
			for i,l in enumerate(self.config.hidden):
				x = tf.layers.batch_normalization(x,training=is_training)
				x = tf.layers.dense(x, 
			                units=l,
			                activation=tf.nn.relu,
			                kernel_initializer=tf.glorot_uniform_initializer(),
			                )
			x = tf.layers.batch_normalization(x,training=is_training)
			q = tf.layers.dense(x,
							units=self.config.rspace,
							activation=None,
							kernel_initializer=self.config.init)
		return q
	def loss(self,y,q):
		return tf.reduce_mean(tf.square(y-q))


class Actor(Base):
	def __init__(self,config,name):
		self.name = name
		super().__init__(config,self.name)
	def _build_network(self,state,is_training):
		with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
			x = state
			for i,l in enumerate(self.config.hidden):
				x = tf.layers.batch_normalization(x,training=is_training)
				x = tf.layers.dense(x, 
			                units=l,
			                activation=tf.nn.relu,
			                kernel_initializer=tf.glorot_uniform_initializer(),
			                )
			x = tf.layers.batch_normalization(x,training=is_training)
			self.mu = tf.layers.dense(x,
							units=self.config.action_dim,
							activation=tf.nn.tanh,
							kernel_initializer=self.config.init)
		return self.mu
	def loss(self, critic):
		return -tf.reduce_mean(critic)

def main():
	env = gym.make("Pendulum-v0")
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	buffer_size = 50000
	batch_size = 128
	num_episodes = 10
	init_frames = 10000
	num_frames = 10000
	eval_frames = 300
	N = 1

	ou = OU(env.action_space)
	reset_ou = ou.reset()
	update_ou = ou.sample()
	replay_buffer = ReplayBuffer(buffer_size)

	critic_config = Config(action_dim,hidden=[128, 64], lr=1e-3)
	actor_config = Config(action_dim,hidden=[64, 32], lr=1e-4,l2=0.0)
	state = tf.placeholder(tf.float32, [None, state_dim], name='state')
	new_state = tf.placeholder(tf.float32, [None,state_dim], name='new_state')
	action = tf.placeholder(tf.float32, [None, action_dim], name='action')
	reward = tf.placeholder(tf.float32, [None,1], name='reward')
	done = tf.placeholder(tf.bool, [None,1], name='done')
	target_Q_vals = tf.placeholder(tf.float32, [None,1], name='target_Q_vals')
	Q = Critic(critic_config,'Critic')
	mu = Actor(actor_config,'Actor')
	target_Q = Critic(critic_config,'Target_Critic')
	target_mu = Actor(actor_config,'Target_Actor')

	is_training = tf.placeholder(tf.bool, (), name='is_training')

	action_mu = mu._build_network(state,is_training)
	value = Q._build_network(tf.concat([state,action],axis=1),is_training)
	value_with_actor = Q._build_network(tf.concat([state, action_mu],axis=1), is_training)
	target_act = target_mu._build_network(new_state,is_training)
	target_val = target_Q._build_network(tf.concat([new_state,target_act],axis=1),is_training)
	target_Q_copy = target_Q.copy_to_target(Q)
	target_mu_copy = target_mu.copy_to_target(mu)

	y = reward + critic_config.gamma**N * target_val * (1.0-tf.cast(done,tf.float32))
	value_loss = Q.loss(target_Q_vals,value)
	actor_loss = mu.loss(value_with_actor)
	value_train = Q.train(value_loss)
	actor_train = mu.train(actor_loss)
	target_value_update = target_Q.update_target(Q)
	target_actor_update = target_mu.update_target(mu)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run([target_Q_copy,target_mu_copy])
		s0 = env.reset()
		s0 = s0.reshape((1,state_dim))
		for i in range(init_frames):
			sinit = None
			ainit = None
			r = 0
			factor = 1
			d = 0
			for j in range(N):
				aval = sess.run(action_mu,feed_dict={state: s0, 
									is_training: False})
				aval += 0.25 * np.random.randn(1,action_dim)
				a = np.clip(aval*env.action_space.high, env.action_space.low, env.action_space.high)
				if not j:
					sinit = s0
					ainit = a
				s1, r0, d, _ = env.step(a)
				s1 = s1.reshape((1,state_dim))
				r += factor * r0
				if d:
					replay_buffer.push(sinit,ainit,r,s1,d)
					s0 = env.reset()
					s0 = s0.reshape((1,state_dim))
					break
				else:
					factor *= critic_config.gamma
					s0 = s1
			if not d:
				replay_buffer.push(sinit, ainit, r, s1, d)
				s0 = s1
		for e in range(num_episodes):
			s0 = env.reset()
			s0 = s0.reshape((1,state_dim))
			sess.run(reset_ou)
			rrew = 0
			rQ_loss = 0
			rmu_loss = 0
			Q_loss_list = [0]
			mu_loss_list = [0]
			for t in range(num_frames):
				sinit = None
				ainit = None
				r = 0
				factor = 1
				d = 0
				for j in range(N):
					aval = sess.run(action_mu,feed_dict={state: s0, 
									is_training: False})
					aval += 0.25 * np.random.randn(1,action_dim)
					a = np.clip(aval*env.action_space.high, env.action_space.low, env.action_space.high)
					if not j:
						sinit = s0
						ainit = a
					s1, r0, d, _ = env.step(a)
					s1 = s1.reshape((1,state_dim))
					r += factor * r0
					if d:
						replay_buffer.push(sinit,ainit,r,s1,d)
						s0 = env.reset()
						s0 = s0.reshape((1,state_dim))
						break
					else:
						factor *= critic_config.gamma
						s0 = s1
				if not d:
					replay_buffer.push(sinit, ainit, r, s1, d)
					s0 = s1
				sval, aval, rval, nsval, dval = replay_buffer.get(batch_size)
				rrew = r0 / (t+1) + t / (t+1) * rrew
				if not t % 1000:
					print("The running reward is ", rrew, " with Q loss ", Q_loss_list[-1], " and mu loss ", mu_loss_list[-1], " for time step ", t, " and episode ", e)
				tQvals = sess.run(y, feed_dict={new_state: nsval,
											    reward: rval,
												done: dval,
												is_training: False})
				if not t % 1000:
					print(np.mean(rval), np.mean(tQvals))
				_, Q_loss = sess.run([value_train, value_loss], 
										feed_dict={state: sval, 
												   target_Q_vals: tQvals,
												   action: aval.reshape((batch_size,action_dim)),
												   is_training: True})
				_, mu_loss = sess.run([actor_train, actor_loss],
										feed_dict={state: sval,
												   is_training: True})
				Q_loss_list.append(Q_loss)
				rQ_loss = Q_loss / (t+1) + t / (t+1) * rQ_loss
				mu_loss_list.append(mu_loss)
				rmu_loss = mu_loss / (t+1) + t / (t+1) * rmu_loss
				sess.run(target_value_update)
				sess.run(target_actor_update)
			s0 = env.reset()
			render = True
			ravg = 0
			for t in range(eval_frames):
				a = sess.run(action_mu, feed_dict={state: s0.reshape((1,state_dim)), is_training: False})
				if render:
					env.render()
				s1, r0, d, _ = env.step(a * env.action_space.high)
				if d:
					s0 = env.reset()
					s0 = s0.reshape((1,state_dim))
				else:
					s0 = s1
				ravg += r0
			ravg /= eval_frames
			print("The reward after episode ", e, " is ", ravg)

if __name__ == '__main__':
	main()