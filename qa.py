from qa_engine.base import QABase
import spacy
import nltk
from nltk.corpus import stopwords
import copy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import sys
import time
import numpy, scipy
from nltk.util import ngrams
import json
import re

nlp = spacy.load("en_core_web_lg")
#test
questions = 0

stop = set(stopwords.words('english'))

def  A6_sentence_selection(question, story):
    q_lemmas = normalize_set(question["question"], expand_synsets=True)
    answers = {

        sent["sentenceid"]: normalize_set(sent["sentence"], expand_synsets=True)
        for sent in story

    }
    for sent in story:
        print(normalize_set(sent["sentence"], expand_synsets=True))
    score_dict = {}
    for ans in answers.keys():
        ans_lemmas = answers[ans]
        score_dict[ans] = len([lem for lem in q_lemmas if lem in ans_lemmas])
    sent_id = max(score_dict, key=score_dict.get)

    threshold = 0
    if score_dict[sent_id] < threshold:
        q_vector = doc_vector(question["question"])
        for sent in story:
            sent_vector = doc_vector(sent["sentence"])
            score_dict[sent["sentenceid"]] = scipy.spatial.distance.cosine(
                q_vector, sent_vector)

        sent_id = min(score_dict, key=score_dict.get)
        threshold2 = 0.5
        if score_dict[sent_id] > threshold2:
            sent_id = "-"

    global questions
    questions += 1
    sys.stdout.write("\r" + str(questions) + " questions  answered...")

    return sent_id, ""
def token_filter(token, use_spacy_stopwords=False):
    keep_token = True
    if token.pos_ == "PROPN":
        keep_token = True
    if token.text == "-PRON-":
        keep_token = True
    elif token.text.lower(
    ) in stop if not use_spacy_stopwords else token.is_stop:
        keep_token = False

    elif token.is_punct:
        keep_token = False
    return keep_token
def normalize(text, lemmatize=True, expand_synsets=False, loose_filter=False):
    doc = nlp(text)
    out = ""
    if expand_synsets and not lemmatize:
        print("can't expand synsets without lemmatizing", file=sys.stderr)
        exit()
    for token in doc:
            if token_filter(token):
                if expand_synsets:
                    pos_dict = {
                        "NOUN": wn.NOUN,
                        "VERB": wn.VERB,
                        "ADV": wn.ADV,
                        "ADJ": wn.ADJ
                    }
                    if token.pos_ in pos_dict:
                        synset = wn.synsets(token.lemma_, pos=pos_dict[token.pos_])
                    else:
                        synset = wn.synsets(token.lemma_)

                    if synset != []:
                        for synonym in synset[0].lemma_names():
                            out += (synonym + " ")
                            for syn in wn.synsets(synonym)[0].lemma_names():
                                out += (syn + " ")
                                if len(wn.synsets(synonym)) > 1:
                                    for syn2 in wn.synsets(synonym)[1].lemma_names():
                                        out += (syn2 + " ")
                        if len(synset) > 1:
                            for synonym in synset[1].lemma_names():
                                out += (synonym + " ")
                    else:
                        out += (token.lemma_ + " ")
                elif (lemmatize):
                    out += ((token.lemma_
                             if token.lemma_ != "-PRON-" else token.text) + " ")
                else:
                    out += (token.text + " ")
    return nlp(out)
def normalize_set(text, lemmatize=True, expand_synsets=False, loose_filter=False):
    doc = nlp(text)
    out_set = set()

    if expand_synsets and not lemmatize:
        print("can't expand synsets without lemmatizing", file=sys.stderr)
        exit()
    for token in doc:
        if token_filter(token):
            if expand_synsets:
                pos_dict = {
                    "NOUN": wn.NOUN,
                    "VERB": wn.VERB,
                    "ADV": wn.ADV,
                    "ADJ": wn.ADJ
                }
                if token.pos_ in pos_dict:
                    synset = wn.synsets(token.lemma_, pos=pos_dict[token.pos_])
                else:
                    synset = wn.synsets(token.lemma_)

                if synset != []:
                    for synonym in synset[0].lemma_names():
                        out_set.add(synonym)
                        for syn in wn.synsets(synonym)[0].lemma_names():
                            out_set.add(syn)
                            if len(wn.synsets(synonym)) > 1:
                                for syn2 in wn.synsets(synonym)[1].lemma_names():
                                    out_set.add(synonym)
                    if len(synset) > 1:
                        for synonym in synset[1].lemma_names():
                            out_set.add(synonym)


                else:
                    out_set.add(token.lemma_)
            elif (lemmatize):
                out_set.add(token.lemma_)
            else:
                out_set.add(token.lemma_)
    return out_set
def doc_vector(text):
    doc = normalize(text)
    docvec = numpy.zeros((300,), dtype="float32")
    for token in doc:
        if token.has_vector:
            docvec = numpy.add(docvec, token.vector)
    # print(docvec.dtype)
    return docvec



SAVED_COREF=("a",dict())
def coreference_story(story):
    ### coreference is incomplete due to cofreference overlap give in JSON ###
    global SAVED_COREF
    if story[0]["storyid"] == SAVED_COREF[0]:
        # print("same story")
        return SAVED_COREF[1]
    else:
        # print("new story")
        # print("question")
        sent = story[0]
        story_coref = sent['coref']
        localstory = copy.deepcopy(story)
        for i in range(0, len(story)):
            # print(story[i]["sentence"])
            localstory[i]["sentence"] = nltk.word_tokenize(
                story[i]["sentence"])
            # print(localstory[i]["sentence"])

        for key in story_coref:
            # print("new antecedent")
            antecedent = story_coref[key][0]["text"]
            for z in range(1, len(story_coref[key])):
                sentind = story_coref[key][z]["sentNum"]
                example_text = story_coref[key][z]["text"]
                antecedent_tokens = nltk.word_tokenize(antecedent)
                # print("antecedent_tokens:", antecedent_tokens)
                example_tokens = nltk.word_tokenize(example_text)
                # print("example_tokens:", example_tokens)
                if example_tokens[0] in localstory[sentind - 1]["sentence"]:
                    example_start = localstory[sentind - 1]["sentence"].index(
                        example_tokens[0])
                    if localstory[sentind - 1]["sentence"][example_start -
                                                           1] != 'was':
                        if localstory[sentind - 1]["sentence"][example_start -
                                                               1] != 'is':
                            if localstory[sentind - 1]["sentence"][
                                    example_start - 2] != antecedent_tokens[
                                        len(antecedent_tokens) - 1]:
                                # print("Before:",
                                #   localstory[sentind - 1]["sentence"])
                                for i in range(0, len(example_tokens)):
                                    del localstory[
                                        sentind - 1]["sentence"][example_start]
                                # print("During:",
                                #       localstory[sentind - 1]["sentence"])
                                for i in range(0, len(antecedent_tokens)):
                                    localstory[sentind - 1]["sentence"].insert(
                                        example_start + i,
                                        antecedent_tokens[i])
                                # print("After:",
                                #       localstory[sentind - 1]["sentence"])
        for i in range(0, len(localstory)):
            localstory[i]["sentence"]=" ".join(localstory[i]["sentence"])
        SAVED_COREF = (story[0]["storyid"], localstory)
        return localstory

'''
def person_in_the_question(question):
  ner_list=question[ner]
  for entry in ner_list:
    if (entry[ner] == 'PERSON') or (entry[ner] == 'TITLE'):
      if(entry[text]!= ("he" or "He" or "she" or "She" or "his" or "hers" or "His" or"Hers"):
        output = entry[text]
        break
  return output
'''
'''def narrow_sentences_by_Who_Where(coref_story, question)
  if nltk.word_tokenize(question["question"])[0]== "Where":
    senetence_list= []
  for sentnce in coref_story:
 '''


def get_answer(question, story):
    """
    :param question: dict
    :param story: dict
    :return: answerid, answer


    question is a dictionary with keys:
        question -- The raw text of question.
        storyid --  The story id.
        questionid  --  The id of the question.
        tokens -- Stanford CoreNLP version of word tokenization with POS
        ner -- Stanford CoreNLP version of NER


    story is a list of sentence, each sentence is a dictionary with keys:
        storytitle -- the title of the story.
        storyid --  the id of the story.
        sentence -- the raw text sentence version.
        sentenceid --  the id of the sentence
        tokens -- Stanford CoreNLP version of word tokenization with POS
        ner -- Stanford CoreNLP version of NER
        coref -- Stanford CoreNLP version of coreference resolution of the entire story

    """

    ###     Your Code Goes Here         ###
    coref_story = coreference_story(story)
    answerid, answer = A6_sentence_selection(question,coref_story)
    # person_in_the_question= person_in_the_question(question)
    # sentences = narrow_sentences_by_Who(coref_story, question)


    ###     End of Your Code         ###
    return answerid, answer


# Code from https://stackoverflow.com/questions/39100652/python-chunking-others-than-noun-phrases-e-g-prepositional-using-spacy-etc
def get_pps(doc):
    "Function to get PPs from a parsed document."
    pps = []
    for token in doc:
        # Try this with other parts of speech for different subtrees.
        if token.pos_ == 'ADP':
            pp = ' '.join([tok.orth_ for tok in token.subtree])
            pps.append(pp)
    return pps


def possible_answers(story, n=3):
    ans = {}
    text = ""
    sentences = []
    for sent in story:
        sentences.append(sent["sentence"])
        text += sent["sentence"] + " "
    doc = nlp(text)
    ans["tokens"] = [token.text for token in doc]
    ans["chunks"] = [chunk.text for chunk in doc.noun_chunks]
    ans["ents"] = {ent.text: ent.label_ for ent in doc.ents}
    for label in set(ans["ents"].values()):
        ans[label] = [key for key in ans["ents"] if ans["ents"][key] == label]
    ans["prep_phrases"] = get_pps(doc)
    ans["ngrams"] = []
    for sentence in sentences:
        ans["ngrams"] += ngrams(nltk.word_tokenize(sentence), n)
    return ans


def question_class(question):
    tokens = nltk.word_tokenize(question["question"].lower())
    question_words = ["is", "was", "does", "did", "had", "when", "what", "where", "who", "how", "why", "which"]
    if tokens[0] not in question_words:
        for token in tokens:
            if token in question_words:
                tokens[0] = token
                break

    if tokens[0] in ["is", "was", "does", "did", "had"]:
        print(question["question"])
        tokens[0] = "yn"
    # if tokens[0] in question_classes.keys():
    #     question_classes[tokens[0]] += 1
    # else:
    #     question_classes[tokens[0]] = 1
    return tokens[0]


#############################################################
###     Dont change the code in this section
#############################################################
class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answerid, answer = get_answer(question, story)
        return (answerid, answer)


def run_qa():
    QA = QAEngine()
    QA.run()
    QA.save_answers()


#############################################################


def main():
    run_qa()


if __name__ == "__main__":
    main()
