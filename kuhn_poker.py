#CFR for Kuhn Poker
import random

PASS = 0
BET = 1
num_actions = 2
actions = ['0','1']

def calculate_pot(k, n):
	pot = [1 for _ in range(n)]
	i = 0
	while k > 0:
		if k&1:
			pot[i] += 1
		k = k >> 1
		i += 1
	pot.reverse()
	return pot

#p = 0, b = 1, ppp = 0, bpp = 4, bpb = 5, bbp = 6, bbb = 7
def terminal_ends(n):
	ends = {}
	ends[0] = [1 for _ in range(n)]
	for i in range(2**(n-1),2**n):
		ends[i] = calculate_pot(i,n)
	return ends

class KuhnNode:
	def __init__(self, infoSet):
		self.infoSet = infoSet
		self.regretSum = [0, 0]
		self.strategy = [0, 0]
		self.strategySum = [0, 0]
		self.parent = []
		self.children = []

	def getStrategy(self, realizationWeight):
		norm = 0
		for a in range(num_actions):
			if self.regretSum[a] > 0: 
				self.strategy[a] = self.regretSum[a]
			else:
				self.strategy[a] = 0
			norm += self.strategy[a]

		for a in range(num_actions):
			if norm > 0:
				self.strategy[a] /= norm
			else:
				self.strategy[a] = 1.0 / num_actions
			self.strategySum[a] += realizationWeight * self.strategy[a]
		return self.strategy

	def getAverageStrategy(self):
		avgStrategy = [0, 0]
		norm = 0
		for a in range(num_actions):
			norm += self.strategySum[a]
		for a in range(num_actions):
			if norm > 0:
				avgStrategy[a] = self.strategySum[a] / norm
			else:
				avgStrategy[a] = 1.0 /num_actions
		return avgStrategy

	def toString(self):
		return self.infoSet + ''.join([a for a in self.getAverageStrategy()])

def argmax(l):
	m = 0
	ind = 0
	n = len(l)
	for i in range(n):
		if m < l[i]:
			m = l[i]
			ind = i
	return ind

def calculate_payout(pot, cards, n):
	bets = []
	payout = [-1 for _ in range(n)]
	for i in range(n):
		if pot[i] > 1:
			bets.append(i)
			payout[i] -= 1
	if not bets:
		payout[argmax(cards[:-1])] = n-1
	else:
		betcards = [0 for _ in range(n)]
		for i in bets:
			betcards[i] = cards[i]
		payout[argmax(betcards)] = n-1+len(bets)-1
	return payout

def permute_pot(pot, history):
	index = 0
	n = len(history)
	while index < n and history[index] != '1':
		index += 1
	return [pot[i-index] for i in range(len(pot))]

def terminal(plays, history, cards, n, ends):
	if plays < n:
		return []
	else:
		if int(history,2) in ends:
			pot = permute_pot(ends[int(history,2)], history)
			return calculate_payout(pot, cards, n)
		else:
			return []

def shuffle(cards):
	for i in range(len(cards)-1, 0, -1):
		j = random.randint(0, i)
		tmp = cards[i]
		cards[i] = cards[j]
		cards[j] = tmp
	return cards

def prod(probabilities, player):
	product = 1
	for i,p in enumerate(probabilities):
		if i != player:
			product *= p
	return product

def cfr(cards, history, p, nodes, n, plays, ends):
	plays += 1
	player = plays % n
	payoff = terminal(plays, history, cards, n, ends)
	if payoff:
		return payoff
	infoSet = str(cards[player]) + history
	if infoSet not in nodes:
		nodes[infoSet] = KuhnNode(infoSet)
	strategy = nodes[infoSet].getStrategy(p[player])
	util = [0 for _ in range(num_actions)]
	nodeUtil = [0 for _ in range(n)]
	for a in range(num_actions):
		nextHistory = history + actions[a]
		nextP = p[:]
		nextP[player] *= strategy[a]
		tree_util = cfr(cards, nextHistory, nextP, nodes, n, plays, ends)
		util[a] = tree_util[player]
		nodeUtil = [n+strategy[a]*u for n,u in zip(nodeUtil, tree_util)]
	for a in range(num_actions):
		regret = util[a] - nodeUtil[player]
		nodes[infoSet].regretSum[a] += prod(p,player) * regret
	return nodeUtil


def train(iterations):
	n_players = 3
	cards = [i+1 for i in range(n_players)]
	cards.append(n_players+1)
	util = [0 for _ in range(n_players)]
	nodes = {}
	ends = terminal_ends(n_players)
	for i in range(iterations):
		shuffle(cards)
		tmp_util = cfr(cards, "", [1 for _ in range(n_players)], nodes, n_players, -1, ends)
		util = [u + t for u,t in zip(util,tmp_util)]
	outputs = []
	for infoSet, node in nodes.items():
		outputs.append([infoSet,node.getAverageStrategy()])
	outputs = sorted(outputs, key=lambda a:a[0])
	for output in outputs:
		print("The average strategy at node", output[0], "is pass probability =", output[1][0], "and bet probability =", output[1][1])
	for i,u in enumerate(util):
		print("Average game value is for player",i,"is", u / iterations)

train(100000)