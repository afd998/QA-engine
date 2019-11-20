from qa_engine.base import QABase
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import copy
import pickle
import sys
import json

nlp = spacy.load("en_core_web_lg")
# test
questions = 0

stop = set(stopwords.words('english'))


def token_filter(token,
                 doc,
                 take_out_capitals=False,
                 use_spacy_stopwords=False):
    keep_token = True
    if token.pos_ == "PROPN":
        keep_token = False
    elif take_out_capitals and (token.text is not doc[0].text) and (
            token.text is not token.text.lower()):
        keep_token is False
    if token.text == "-PRON-":
        keep_token = False
    elif token.text.lower(
    ) in stop if not use_spacy_stopwords else token.is_stop:
        keep_token = False

    elif token.is_punct:
        keep_token = False
    return keep_token


def expand_synsets(token):
    out_set = set()
    pos_dict = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADV": wn.ADV, "ADJ": wn.ADJ}
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
    return out_set


def normalize(text,
              output_type="set",
              take_out_caps=False,
              lemmatize=True,
              expand_synsets=True,
              loose_filter=False):
    if expand_synsets and not lemmatize:
        print("can't expand synsets without lemmatizing", file=sys.stderr)
        exit()

    doc = nlp(text)
    for token in doc:
        if token_filter(token, doc, take_out_capitals=take_out_caps):
            if expand_synsets:
                expand_synsets(token)
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
                                for syn2 in wn.synsets(
                                        synonym)[1].lemma_names():
                                    out_set.add(synonym)
                    if len(synset) > 1:
                        for synonym in synset[1].lemma_names():
                            out_set.add(synonym)
                else:
                    out_set.add(token.lemma_)
            elif (lemmatize):
                out_set.add((
                    token.lemma_ if token.lemma_ != "-PRON-" else token.text) +
                            " ")
            else:
                out_set.add(token.text + " ")
    if output_type == "string":
        list1 = list(out_set)
        out = ''.join(list1)
    else:
        out = out_set
    return nlp(out)


def coreference_story(story, load):
    ### coreference is incomplete due to coreference overlap give in JSON ###
    if not load:
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
                                #   localstory[sentind - 1]["sentence"])
                                for i in range(0, len(antecedent_tokens)):
                                    localstory[sentind - 1]["sentence"].insert(
                                        example_start + i,
                                        antecedent_tokens[i])
                                # print("After:",
                                #   localstory[sentind - 1]["sentence"])
        coref_file = open(
            "data/coref_sents-{0}.pickle".format(story[0]["storyid"]), 'wb')
        pickle.dump(localstory, coref_file)
        coref_file.close()
        return localstory
    else:
        coref_file = open(
            "data/coref_sents-{0}.pickle".format(story[0]["storyid"]), 'rb')
        localstory = pickle.load(coref_file)
        coref_file.close()
        return localstory


'''
def person_in_the_question(question):
  ner_list=question[ner]
  for entry in ner_list:
    if (entry[ner] == 'PERSON') or (entry[ner] == 'TITLE'):
      if(entry[text].lower() not in ["he", "she", "his", "hers"]:
        output = entry[text]
        break
  return output
'''
'''def narrow_sentences_by_Who_Where(coref_story, question)
  if nltk.word_tokenize(question["question"])[0]== "Where":
    senetence_list= []
  for sentnce in coref_story:
 '''

# print_story = True

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
    # global print_story
    # if print_story:
    #     print_story = False
    #     print(json.dumps(story, indent=4))
    load = False
    coref_story = coreference_story(story, load)
    question_class(question)

    # person_in_the_question= person_in_the_question(question)
    # sentences = narrow_sentences_by_Who(coref_story, question)

    answer = "whatever you think the answer is"
    answerid = "-"

    return answerid, answer

# def possible_answers(story):
question_classes = {}
def question_class(question):
    tokens = nltk.word_tokenize(question["question"])
    if tokens[0].lower() in ["is", "was", "does", "did", "had"]:
        print(question["question"])
        tokens[0] = "yn"
    if tokens[0].lower() in question_classes.keys():
        question_classes[tokens[0].lower()] += 1
    else:
        question_classes[tokens[0].lower()] = 1
    return tokens[0].lower()

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
    print(question_classes)


#############################################################


def main():
    run_qa()


if __name__ == "__main__":
    main()
