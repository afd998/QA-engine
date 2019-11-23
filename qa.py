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


nlp = spacy.load("en_core_web_lg")
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
    ans = possible_answers(question, story)
    hq = head_of_question(question, story)
    # print(hq)
    triple = triple_check(hq, story)
    triple_answer = None
    if q_class in ["who", "what"] and triple is not None and triple is not ():
        if triple[2] != "" and triple[2] not in question["question"]:
            triple_answer = triple[2]
        elif triple[0] != "" and triple[0] not in question["question"]:
            triple_answer = triple[0]
        elif triple[1] != "" and triple[1] not in question["question"] and "do" in question["question"]:
            triple_answer = triple[1]
    if triple_answer is not None:
        answer = triple_answer
    elif q_class in ["how", "why"]:
        a6ID, a6ans = A6_sentence_selection(question, story)
        answer = [sentence["sentence"] for sentence in story if sentence["sentenceid"] == a6ID][0]
    else:
        time_prepositions = ["after", "before", "during", "while"]
        if q_class == "yn":
            answer = "yes no"
        else:
            if q_class == "who":
                possible = [
                    ent for ent in ans["ents"]
                    if ent[1].label_ in ["PERSON", "ORG"]
                ]
            elif q_class == "what":
                possible = ans["chunks"]
            elif q_class == "when":
                possible = []
                possible = [
                    pp for pp in ans["pps"]
                    if pp[1][0].text.lower() in time_prepositions
                ] + [
                    ent for ent in ans["ents"]
                    if ent[1].label_ in ["TIME", "CARDINAL", "DATE"]
                ]
            elif q_class == "where":
                possible = [
                    pp for pp in ans["pps"]
                    if pp[1][0].text.lower() not in time_prepositions
                ]
            if len(possible) == 0:
                possible = ans["chunks"]
            answer = best_answer(question, possible, keyword = hq)
    ###     End of Your Code         ###
    answerid = "-"
    print(answer)
    return answerid, answer


def best_answer(question, answers, keyword=None):
    q_bag = bag_words(nlp(question["question"]))
    score_dict = {}
    for ans in answers:
        front_bag = bag_words(ans[0])
        back_bag = bag_words(ans[2])
        front_score = len([x for x in q_bag if x in front_bag])
        back_score = len([x for x in q_bag if x in back_bag])
        score_dict[ans[1].text] = max(front_score, back_score)
    return max(score_dict, key=score_dict.get)


def possible_answers(question, story):
    question = question["question"]
    q_doc = nlp(question.replace('"', ""))
    q_spans = set([chunk.text for chunk in q_doc.noun_chunks] +
                  [ent.text
                   for ent in q_doc.ents] + [pp.text for pp in get_pps(q_doc)])
    # question_nonstop_set = set([chunk.text for token in q_doc if token not in stop])
    sentences = [sent["sentence"].replace('"', "") for sent in story]
    text = "\n".join(sentences)
    ans = {}
    doc = nlp(text)
    ans["chunks"] = []
    ans["ents"] = []
    ans["pps"] = []
    ents_set = set([])
    for ent in doc.ents:
        if ent not in q_spans and ent.text.strip() != "":
            ents_set.add(ent.text)
            front_context, back_context = span_context(ent)
            ans["ents"].append((front_context, ent, back_context))
    for chunk in doc.noun_chunks:
        if chunk.text not in ents_set and chunk.text not in q_spans and chunk.text.lower(
        ) not in stop and chunk.text.strip() != "":
            front_context, back_context = span_context(chunk)
            ans["chunks"].append((front_context, chunk, back_context))
    for pp in get_pps(doc):
        if pp not in q_spans and pp.text.strip() != "":
            front_context, back_context = span_context(pp)
            ans["pps"].append((front_context, pp, back_context))
    return (ans)


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


def span_context(span, context_size=7):
    doc = span.doc
    context_start = max(0, span.start - context_size)
    context_end = min(span.end + context_size, len(doc))
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
    if tokens[0] == "which":
        tokens[0] = "what"
    # if tokens[0] in question_classes.keys():
    #     question_classes[tokens[0]] += 1
    # else:
    #     question_classes[tokens[0]] = 1
    # #print(tokens[0])
    return tokens[0]


def bag_words(doc, lemmatize=True):
    bag = set([])
    for token in doc:
        if token.text.lower() not in stop:
            bag.add(token.lemma_ if lemmatize else token.text.lower())
    return bag

def triple_check(keyword, story):
    sentences = [sent["sentence"].replace('"', "") for sent in story]
    doc = nlp("\n".join(sentences))
    if isinstance(keyword, spacy.tokens.span.Span):
        keyword = keyword.root
    #print(keyword)
    if check_if_in_story(keyword, story):
        keyword = find_in_story(keyword, story)
        # print(keyword)
    else:
        return None
    headword = keyword.head
    obj = ""
    subject = ""
    for child in headword.children:
        if "subj" in child.dep_:
            subject = (child)
        if child.dep_ == "attr":
            obj = " ".join([token.text for token in child.subtree])
        elif "obj" in child.dep_:
            obj = " ".join([token.text for token in child.subtree])
        # elif "prep" in child.dep_:
        #     objects.append(" ".join([token.text for token in child.subtree]))
    return((str(subject), str(headword), str(obj)))

def check_if_in_story(token, story):
    text = ""
    sentences = []
    for sent in story:
        sentences.append(sent["sentence"])
        text += sent["sentence"] + " "
    doc = nlp(text)
    for word in doc:
        if word.lemma_ == token.lemma_:
            return True
    return False


def find_in_story(token, story):
    text = ""
    sentences = []
    for sent in story:
        sentences.append(sent["sentence"])
        text += sent["sentence"] + " "
    doc = nlp(text)
    for word in doc:
        if word.lemma_ == token.lemma_:
            return word


def head_of_question(question, story):
    doc = nlp(question["question"])
    if question_class(question) in ["who"]:
        #print("QUESTION:", question["question"])
        tok = doc[0]
        while (tok != tok.head):
            tok = tok.head

        if tok.is_stop or not check_if_in_story(tok, story):
            chunks = [chunk for chunk in doc.noun_chunks]
            if len(chunks)>1:
                for chunk in reversed(chunks):
                    if chunk.text not in stop:
                        if chunk.text[0].lower() == chunk.text[0]:
                            #print(chunk.text)
                            return chunk
            if "ADJ" in [token.pos_ for token in doc]:
                for token in doc:
                    if token.pos_ == "ADJ":
                        #print(token.text)
                        return token

            else:
                maxtoken=doc[0]
                for token in doc:
                    if len(token.text)>=len(maxtoken.text):
                        maxtoken = token
                #print(maxtoken)
                return maxtoken
        else:
            #print(tok.text)
            return tok

    elif question_class(question) in ["what", "which"]:
        #print("QUESTION:", question["question"])
        tok = doc[0]
        while (tok != tok.head):
            tok = tok.head
        if "ADJ" in [token.pos_ for token in doc]:
            for token in doc:
                if token.pos_ == "ADJ":
                    #print(token.text)
                    return token
        elif tok.text in stop:
            chunks = [chunk for chunk in doc.noun_chunks]
            if len(chunks) > 1:
                for chunk in reversed(chunks):
                    if chunk.text not in stop:
                        #print(chunk.text)
                        return chunk


            else:
                maxtoken = doc[0]
                for token in doc:
                    if len(token.text) >= len(maxtoken.text):
                        maxtoken = token
                #print(maxtoken)
                return maxtoken
        else:
            #print(tok.text)
            return tok

    elif question_class(question) in ["when"]:
        #print("QUESTION:", question["question"])
        chunks = [chunk for chunk in doc.noun_chunks]
        for chunk in chunks:
            if " " in chunk.text:
                #print(chunk.text)
                return chunk
            for token in doc:
                if token.tag_ == "ADJ":
                    #print(token.text)
                    return token
            maxtoken = doc[0]
            for token in doc:
                if len(token.text) >= len(maxtoken.text):
                    maxtoken = token
            #print(maxtoken)
            return maxtoken

    elif question_class(question) in ["why", "how"]:
        #print("QUESTION:", question["question"])
        tok = doc[len(doc)-1]
        while (tok != tok.head):
            tok = tok.head
        if tok in stop:
            for token in reversed(doc):
                if token.tag_ in ["VB", "VBG" "VBN", "VBP", "VBD", "VBZ"]:
                    #print(token.text)
                    return token
            for token in reversed(doc):
                if token.tag_ == "ADJ":
                    #print(token.text)
                    return token
            maxtoken = doc[0]
            for token in doc:
                if len(token.text) >= len(maxtoken.text):
                    maxtoken = token
            #print(maxtoken)
            return maxtoken
        #print(tok.text)
        return tok

    elif question_class(question) in ["where"]:
        #print("QUESTION:", question["question"])
        tok = doc[len(doc)-1]
        while (tok != tok.head):
            tok = tok.head
        if tok.text in stop:
            for token in reversed(doc):
                if token.tag_ in ["VB", "VBG" "VBN", "VBP", "VBD", "VBZ"]:
                    #print(token.text)
                    return token
            for token in reversed(doc):
                if token.tag_ == "ADJ":
                    #print(token.text)
                    return token
            maxtoken = doc[0]
            for token in doc:
                if len(token.text) >= len(maxtoken.text):
                    maxtoken = token
            #print(maxtoken)
            return maxtoken
        #print(tok.text)
        return tok

    elif question_class(question) in ["yn"]:
        #print("QUESTION:", question["question"])
        tok = doc[0]
        #print(tok.text)
        return tok

def A6_sentence_selection(question, story):
    q_lemmas = normalize_set(question["question"], expand_synsets=True)
    answers = {

        sent["sentenceid"]: normalize_set(sent["sentence"], expand_synsets=True)
        for sent in story

    }
    # for sent in story:
        # print(normalize_set(sent["sentence"], expand_synsets=True))
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


def normalize_set(text, lemmatize=True, expand_synsets=True, loose_filter=False):
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
