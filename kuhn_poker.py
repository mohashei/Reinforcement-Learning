#CFR for Kuhn Poker
import random

PASS = 0
BET = 1
num_actions = 2
actions = ['p','b']

class Node:
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

def terminal(plays, history, cards, player, opponent):
	if plays > 1:
		terminalPass = history[plays-1] == 'p'
		doubleBet = history[plays-2:] == 'bb'
		isPlayerCardHigher = cards[player] > cards[opponent]
		if terminalPass:
			if history == 'pp':
				if isPlayerCardHigher: return 1
				else: return -1
			else:
				return 1
		elif doubleBet:
			if isPlayerCardHigher: return 2
			else: return -2
		else:
			return 0
	else:
		return 0

def shuffle(cards):
	for i in range(len(cards)-1, 0, -1):
		j = random.randint(0, i)
		tmp = cards[i]
		cards[i] = cards[j]
		cards[j] = tmp
	return cards

def cfr(cards, history, p, nodes):
	plays = len(history)
	player = plays % 2
	opponent = 1 - player
	payoff = terminal(plays, history, cards, player, opponent)
	if payoff:
		return payoff
	infoSet = str(cards[player]) + history
	if infoSet not in nodes:
		nodes[infoSet] = Node(infoSet)
	strategy = nodes[infoSet].getStrategy(p[player])
	util = [0, 0]
	nodeUtil = 0
	for a in range(num_actions):
		nextHistory = history + actions[a]
		nextP = p[:]
		nextP[player] *= strategy[a]
		util[a] = -cfr(cards, nextHistory, nextP, nodes)
		nodeUtil += strategy[a] * util[a]
	for a in range(num_actions):
		regret = util[a] - nodeUtil
		nodes[infoSet].regretSum[a] += p[opponent] * regret
	return nodeUtil


def train(iterations):
	cards = [1, 2, 3]
	util = 0
	nodes = {}
	for i in range(iterations):
		shuffle(cards)
		util += cfr(cards, "", [1, 1], nodes)
	outputs = []
	for infoSet, node in nodes.items():
		outputs.append([infoSet,node.getAverageStrategy()])
	outputs = sorted(outputs, key=lambda a:a[0])
	for output in outputs:
		print("The average strategy at node", output[0], "is pass probability =", output[1][0], "and bet probability =", output[1][1])
	print("Average game value is ", util / iterations)

train(100000)