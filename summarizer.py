import math
from collections import defaultdict
import nltk.data
from nltk import word_tokenize,sent_tokenize
from collections import OrderedDict
import collections
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
from nltk.corpus import brown
import parser
try:
    import matplotlib.pyplot as plt
except:
    raise

import networkx as nx

# create a graph with sentences as its vertices
class Graph:
	def __init__(self):
		self.Vertices = []
		self.Edges = []

	def getRankedVertices(self):
		res = defaultdict(float)
		for e in self.Edges:
			res[e.Vertex1] += e.Weight
		return sorted(res.items(), key=lambda x: x[1], reverse=True)

class Vertex:
	def __init__(self):
		self.Sentence = None

class Edge:
	def __init__(self):
		self.Vertex1 = None
		self.Vertex2 = None
		self.Weight = 0

class Sentence:

	def __init__(self):
		self.s = ""
		self.keywords = []


class Paragraph:
	def __init__(self):
		self.Sentences = []

#################################################################################################################################################
# parser - returns tokenized sentences and keywords(cleaned)
#################################################################################################################################################

# paragraph = ["Mr. Green killed Colonel Mustard in the study with the candlestick. Mr. Green is not a very nice fellow. Professor Plum has a green plant in his study. Miss Scarlett watered Professor Plum's green plant while he was away from his office last week."]

class Parser:
	def read_from_file(self, fname):
		with open(fname) as f:
			content = f.readlines()
		f.close()
		return content

	def sentence_tokenizer(self, para):
		#use tokenizer to tokenize sentences in the sentence
		# sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		# actual_sentences = sent_detector.tokenize(paragraph[0])
		tokenized_sentences = []
		for sentence in para:
			tokenized_sentences.append(sent_tokenize(sentence))

		i = 0
		tokenized_sentences = [item for sublist in tokenized_sentences for item in sublist]
		# for t in tokenized_sentences:
		# 	print str(i) + " " + t 
		# 	i = i + 1
		return tokenized_sentences

	def remove_punctuations(self, text_string):
		#defining a list of special characters to be used for text cleaning
		special_characters = [",",".","'",";","\n", "?", "!", ":", ")", "("] 
		cleaned_string = str(text_string)
		# removing stop words
		for ch in special_characters:
			cleaned_string = cleaned_string.replace(ch, "")
			cleaned_string = cleaned_string.lower()
		return cleaned_string

	def remove_stop_words(self, document):
		# defining a stop word list
		# print document
		document = document[0].split(" ")
		
		words_file = "stop_words.txt"
		stop_word_list = []
		stop_word_list = [word for line in open(words_file, 'r') for word in line.split(",")]
		cleaned_doc = []

		for term in document:
			term = self.remove_punctuations(term)
			if term not in stop_word_list:
				# print term
				cleaned_doc.append(term)
		# print cleaned_doc
		return cleaned_doc     

	def word_tokenizer(self, para):
		# removing stop words and punctuations
		clean_para = self.remove_stop_words(para)
		# print clean_para
		tokenized_words = []
		for w in clean_para:
			tokenized_words.append(word_tokenize(w))

		tokenized_words = [item for sublist in tokenized_words for item in sublist]
		# print tokenized_words
		return tokenized_words

	# ranked_sentences = sentence_ranker(tokenized_sentences)

	# remove the punctuations and stop words from the paragraph and get keywords 



class Reduction:

	def getSentences(self, tokenized_sentences):
		sentences = []

		for t in tokenized_sentences:
			x = []
			x.append(str(t))
			tokens = parser.word_tokenizer(x)
			# print tokens
			sentence = Sentence()
			sentence.s = t
			sentence.keywords = tokens
			sentences.append(sentence)
		
		return sentences

	def findWeight(self, sentence1, sentence2):
		length1 = len(sentence1.s)
		length2 = len(sentence2.s)
		if length1 < 4 or length2 < 4:
			return 0
		weight = 0
		for w1 in sentence1.keywords:
			for w2 in sentence2.keywords:
				if w1 == w2:
					weight = weight + 1
		normalised1 = 0
		if length1 > 0:
			normalised1 = math.log(length1)
		normalised2 = 0
		if length2 > 0:
			normalised2 = math.log(length2)
		norm = normalised1 + normalised2
		if norm == 0:
			return 0
		# print str(sentence1.s) + "\n" + str(sentence2.s) + "\n" + str(weight) + "\n"
		return weight / float(norm)

	def buildGraph(self, sentences):
		g = Graph()
		G=nx.Graph()
		dict = {}
		for s in sentences:
			v = Vertex()
			v.Sentence = s
			g.Vertices.append(v)
			# print v, v.Sentence
			i1 = 1
			j1 = 1
		for i in g.Vertices:
			for j in g.Vertices:
				if i != j:
					w = self.findWeight(i.Sentence, j.Sentence)
					e = Edge()
					if (w != 0):
						dict[i.Sentence.s,j.Sentence.s] = w
					e.Vertex1 = i
					e.Vertex2 = j
					e.Weight = w
					g.Edges.append(e)
				else: 
					w = 0
				G.add_edge(i1,j1,weight=w)
				j1+=1
			i1 += 1
		
		elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.5]
		esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.5]

		pos=nx.spring_layout(G) # positions for all nodes

		# nodes
		nx.draw_networkx_nodes(G,pos,node_size=700)

		# edges
		nx.draw_networkx_edges(G,pos,edgelist=elarge, width=6)
		nx.draw_networkx_edges(G,pos,edgelist=esmall, width=6,alpha=0.5,edge_color='b')

		# labels
		nx.draw_networkx_labels(G,pos,font_size= 8,font_family='sans-serif')

		plt.axis('off')
		plt.savefig("weighted_graph.png") # save as png
		plt.show()
		# print dict
		return g

	def sentenceRank(self, sentences):
		g = self.buildGraph(sentences)
		return g.getRankedVertices()


	def reduce(self, tokenized_sentences, summary_length, GRAPH_SCORE_FACTOR):

		parser = Parser()
		sentences = self.getSentences(tokenized_sentences)
		rankedSentences = self.sentenceRank(sentences)
		orderedSentences = []

		for s in sentences:
			orderedSentences.append(s)

		reducedSentences = []
		i = 0
		while i < summary_length:
			s = rankedSentences[i][0].Sentence
			position = orderedSentences.index(s)
			# print(s,position)
			reducedSentences.append((s, position))
			i = i + 1
		
		reducedSentences = sorted(reducedSentences, key=lambda x: x[1])
		score_from_graph_summary_dict = {}
		score = len(reducedSentences)*GRAPH_SCORE_FACTOR
		# print score
		reducedText = []
		for sentence,rank in reducedSentences:
			score_from_graph_summary_dict[sentence.s] = score 
			score = score - GRAPH_SCORE_FACTOR
			reducedText.append(sentence.s)
		# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
		# print reducedText
		return score_from_graph_summary_dict
	


	
# print words_from_audio_dict
class Summarizer:
	# term frequency - no. of times a word appers in a sentence
	def logarithmic_tf(self, term, sentence):
		count = sentence.count(term)
		if count == 0:
			return 0
		else:
			return 1 + math.log(count)
		# returns a dictionary of all words in the corpus with their logarithmic term frequencies


	def set_freq_to_words(self, para):
		global parser
		words = parser.word_tokenizer(para)
		word_set = set(words)
		word_freq = {}
		for w in word_set:
			count = self.logarithmic_tf(w, words)
			word_freq[w] = count

		word_freq = OrderedDict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))
		# print word_freq
		# returns a dictionary of keywords and corresponding term frequencies
		return word_freq


	def get_audio_word_dict(self, AMPLITUDE_FACTOR):
		global words_from_audio_dict

		audio_word_list = [word for line in open("words_from_audio.txt", 'r') for word in line.split(",")]
		words_from_audio_dict = {}

		for w in audio_word_list:
			words_from_audio_dict[w] = AMPLITUDE_FACTOR

		return words_from_audio_dict
	
	def title_generator(self, tokenized_sentences, word_freq):
		sentences = self.getSentences(tokenized_sentences)
		propernouns=[]
		for s in sentences:
			tagged_sent=pos_tag(s.keywords)
			propernouns.append([word for word,pos in tagged_sent if pos == 'NNP'])
	        
		max=0
		
		for p in propernouns:
			if word_freq[p]>max:
				max=word_freq[p]
				noun=p

		adj=[]
		for s in sentences:
			if noun in s.keywords:
				tagged_sent=pos_tag(s.keywords)
				adj.append([word for word,pos in tagged_sent if pos == 'ADJ'])

		max=0
		for a in adj:
	 		if word_freq[a]>max:
				max=word_freq[p]
				adjective=a

		return adjective+" "+noun
	
	def sentence_ranker(self, sentences, score_from_graph_summary_dict, AMPLITUDE_FACTOR, TF_FACTOR):
		# returns a dictionary of ranked sentences
		# print score_from_graph_summary_dict
		global parser
		ranked_sentences = {}
		for s in sentences:
			score = 0
			# scores of top N sentences from graph based summary
			try:
				score = score + score_from_graph_summary_dict[s]
			except:
				score = score + 0

			# score from term-frequencies of keywords
			for w in parser.remove_punctuations(s).split():
				try:
					score = score + (tf_dictionary[w] * TF_FACTOR)
				except:
					score = score + 0

			# score from audio-retrieved words
			words_from_audio_dict = self.get_audio_word_dict(AMPLITUDE_FACTOR)
			for w in parser.remove_punctuations(s).split():
				try:
					score = score + (words_from_audio_dict[w])
					# print w
				except:
					score = score + 0

			ranked_sentences[s] = score

		ranked_sentences = OrderedDict(sorted(ranked_sentences.items(), key=lambda x: x[1], reverse=True))
		# print ranked_sentences
		return ranked_sentences

	def summarize(self, input, tokenized_sentences, length):
		top_sentences =  dict(collections.Counter(input).most_common(length))
		# print top_sentences
		summary_file = open("summary.txt", "w") 
		# title=self.title_generator(tokenized_sentences, tf_dictionary, )
		# print title
		# summary_file.write(heading)
		# summary_file.write("\n")
		summary = ""
		for s in tokenized_sentences:
			try:		
				if (s in top_sentences.keys()):
					summary += s
					summary += " "
					summary_file.write(s)
					summary_file.write(" ")
			except:
				pass
		# print "summary generated in file..."
		print len(tokenized_sentences)
		summary_file.close()
		return summary
	# get top keywords

	def generate_summary(self, input_file, SUMMARY_LENGTH):
		global GRAPH_SCORE_FACTOR
		global AMPLITUDE_FACTOR
		global TF_FACTOR
		global parser
		# read input from text file, get paragraph for summarization 
		parser = Parser()
		paragraph = parser.read_from_file(input_file)
		# get a dictionary containing term frequencies of keywords
		tf_dictionary = self.set_freq_to_words(paragraph)
		# generate tokenized sentences
		tokenized_sentences = parser.sentence_tokenizer(paragraph)
		# graph based summariztion
		score_from_graph_summary_dict = Reduction().reduce(tokenized_sentences, SUMMARY_LENGTH, GRAPH_SCORE_FACTOR)
		# print score_from_graph_summary_dict
		# assign score to every sentence based of Tf, corresponding amplitude, and graph edges
		ranked_sentences = self.sentence_ranker(tokenized_sentences, score_from_graph_summary_dict, AMPLITUDE_FACTOR, TF_FACTOR)
		# generate summary: get top n sentences from the ranked corpus
		summary = self.summarize(ranked_sentences, tokenized_sentences, SUMMARY_LENGTH)
		return summary

	def set_factors(self, graph, audio, tf):
		global GRAPH_SCORE_FACTOR
		global AMPLITUDE_FACTOR
		global TF_FACTOR
		GRAPH_SCORE_FACTOR = graph
		AMPLITUDE_FACTOR = audio
		TF_FACTOR = tf







