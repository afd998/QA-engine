from qa_engine.base import QABase
import spacy
import nltk
from nltk.corpus import stopwords
import copy
import pickle

nlp = spacy.load("en_core_web_lg")
#test
questions = 0

stop = set(stopwords.words('english'))


def token_filter(token, doc, take_out_capitals=False, use_spacy_stopwords=False):
    keep_token = True
    if token.pos_ == "PROPN":
        keep_token = False
    elif take_out_capitals and (token.text is not doc[0].text) and (token.text is not token.text.lower()):
        keep_token == False
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
    return out_set


def normalize(text, output_type="set", take_out_caps=False, lemmatize=True, expand_synsets=True, loose_filter=False):
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
                                for syn2 in wn.synsets(synonym)[1].lemma_names():
                                    out_set.add(synonym)
                    if len(synset) > 1:
                        for synonym in synset[1].lemma_names():
                            out_set.add(synonym)
                else:
                    out_set.add(token.lemma_)
            elif (lemmatize):
                out_set.add((token.lemma_
                             if token.lemma_ != "-PRON-" else token.text) + " ")
            else:
                out_set.add(token.text + " ")
    if output_type == "string":
        list1 = list(out_set)
        out = ''.join(list1)
    else:
        out = out_set
    return nlp(out)
SAVED_COREF=("a",dict())

def coreferece_story(story):
    ### coreference is incomplete due to cofreference overlap give in JSON ###
    global SAVED_COREF
    if story[0]["storyid"] == SAVED_COREF[0]:
        print("same story")
        return SAVED_COREF[1]
    else:
        print("new story")
        print("question")
        sent = story[0]
        story_coref = sent['coref']
        localstory = copy.deepcopy(story)
        for i in range(0, len(story)):
            # print(story[i]["sentence"])
            localstory[i]["sentence"] = nltk.word_tokenize(story[i]["sentence"])
            # print(localstory[i]["sentence"])

        for key in story_coref:
            print("new anticident")
            anticident = story_coref[key][0]["text"]
            for z in range(1, len(story_coref[key])):
                sentind = story_coref[key][z]["sentNum"]
                example_text = story_coref[key][z]["text"]
                anticident_tokens = nltk.word_tokenize(anticident)
                print("anticident_tokens:", anticident_tokens)
                example_tokens = nltk.word_tokenize(example_text)
                print("example_tokens:", example_tokens)
                if example_tokens[0] in localstory[sentind - 1]["sentence"]:
                    example_start = localstory[sentind - 1]["sentence"].index(example_tokens[0])
                    if localstory[sentind - 1]["sentence"][example_start - 1] != 'was':
                        if localstory[sentind - 1]["sentence"][example_start - 1] != 'is':
                            if localstory[sentind - 1]["sentence"][example_start - 2] != anticident_tokens[
                                len(anticident_tokens) - 1]:
                                print("Before:", localstory[sentind - 1]["sentence"])
                                for i in range(0, len(example_tokens)):
                                    del localstory[sentind - 1]["sentence"][example_start]
                                print("During:", localstory[sentind - 1]["sentence"])
                                for i in range(0, len(anticident_tokens)):
                                    localstory[sentind - 1]["sentence"].insert(example_start + i, anticident_tokens[i])
                                print("After:", localstory[sentind - 1]["sentence"])
        SAVED_COREF= (story[0]["storyid"],localstory )
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
    print("NEW QUESTION")
    coref_story = coreferece_story(story)
    # person_in_the_question= person_in_the_question(question)
    # sentences = narrow_sentences_by_Who(coref_story, question)

    answer = "whatever you think the answer is"
    answerid = "-"

    ###     End of Your Code         ###
    return answerid, answer


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
