from qa_engine.base import QABase
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

from json import dumps

nlp = spacy.load("en_core_web_sm")
stop = set(stopwords.words('english'))

def get_answer(question, story):
    """
    :param question: dict
    :param story: dict
    :return: answerid, answer


    question is a dictionary with keys:
        question -- The raw text of question.
        storyid --  The story id.
        questionid  --  The id of the question.
        ner -- Stanford CoreNLP version of NER


    story is a list of sentence, each sentence is a dictionary with keys:
        storytitle -- the title of the story.
        storyid --  the id of the story.
        sentence -- the raw text sentence version.
        sentenceid --  the id of the sentence
        ner -- Stanford CoreNLP version of NER
        coref -- Stanford CoreNLP version of coreference resolution of the entire story

    """
    ###     Your Code Goes Here         ###
    # global last_story
    # if (last_story != question["storyid"]):
    #     last_story = question["storyid"]
    #     answer_possibilities.append(possible_answers(question, story))
    q_class = question_class(question)
    possibilities = possible_answers(question, story)
    if q_class == "yn":
        answer = "yes no"


    answer = "whatever you think the answer is"
    answerid = "-"
    question_class(question)


    ###     End of Your Code         ###
    return answerid, answer

def possible_answers(question, story):
    question = question["question"]
    q_doc = nlp(question)
    q_spans = set([chunk.text for chunk in q_doc.chunks] + [ent.text for ent in q_doc.ents] + [pp.text for pp in get_pps(q_doc)])
    # question_nonstop_set = set([chunk.text for token in q_doc if token not in stop])
    sentences = [sent["sentence"] for sent in story]
    text = "\n".join(sentences)
    ans = {}
    doc = nlp(text)
    ans["chunks"] = []
    ans["ents"] = []
    ans["pps"] = []
    ents_set = set([])
    for ent in doc.ents:
        if ent not in q_spans:
            ents_set.add(ent.text)
            front_context, back_context = span_context(ent)
            ans["ents"].append((front_context, ent, back_context))
    for chunk in doc.noun_chunks:
        if not any([chunk.text in s for s in [ents_set, q_spans] + chunk.text.lower() in stop]):
            front_context, back_context = span_context(chunk)
            ans["chunks"].append((front_context, chunk, back_context))
    for pp in get_pps(doc):
        if pp not in q_spans:
            front_context, back_context = span_context(pp)
            ans["pps"].append((front_context, pp, back_context))
    return(ans)


def get_pps(doc):
    # adapted from stackoverflow, not entirely my code
    pps = []
    for token in doc:
        if token.pos_ == 'ADP':
            pp_starti = token.i
            for tok in token.subtree:
                pp_endi = tok.i
            pps.append(doc[pp_starti:pp_endi])
    return pps



def span_context(span, context_size = 5):
    doc = span.doc
    context_start = max(0, span.start-context_size)
    context_end = min(span.end+context_size, len(doc))
    front_context = doc[context_start:span.start]
    end_context = doc[span.end:context_end]
    return front_context, end_context

def question_class(question):
    tokens = nltk.word_tokenize(question["question"].lower())
    question_words = [
        "is", "was", "does", "did", "had", "when", "what", "where", "who",
        "how", "why", "which"
    ]
    if tokens[0] not in question_words:
        for token in tokens:
            if token in question_words:
                tokens[0] = token
                break
    if tokens[0] in ["is", "was", "does", "did", "had"]:
        tokens[0] = "yn"
    # if tokens[0] in question_classes.keys():
    #     question_classes[tokens[0]] += 1
    # else:
    #     question_classes[tokens[0]] = 1
    # print(tokens[0])
    return tokens[0]

def bag_words(doc, lemmatize=True):
    bag = set([])
    for token in doc:
        if token.text.lower() not in stop:
            bag.add(token.lemma_ if lemmatize else token.text.lower())
    return bag
# def expand_synsets(bag):
#     syn_bag = set([]):



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
