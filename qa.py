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

nlp = spacy.load("en_core_web_md")
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
    # print(json.dumps(story[0]["coref"], indent=4))
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
    #ATTICUS' who
    if q_class in ["who"]:
        return "-" , extract_who_answer(story, question, 0)
    #JAMES's everything else
    else:
        if q_class in ["who", "what"] and triple is not None and triple is not ():
            # print()
            # print(question["question"])
            # print(triple)
            if triple[2] != "" and triple[2] not in question["question"]:
                triple_answer = triple[2]
            # elif triple[0] != "" and triple[0] not in question["question"]:
            #     triple_answer = triple[0]
            # elif triple[1] != "" and triple[1] not in question["question"] and "do" in question["question"]:
            #     triple_answer = triple[1]
        if triple_answer is not None:
            print(triple_answer)
            answer = triple_answer
        elif q_class in ["how", "why"]:
            a6ID, a6ans = A6_sentence_selection(question, story)
            answer = [sentence["sentence"] for sentence in story if sentence["sentenceid"] == a6ID][0]
        else:
            time_prepositions = ["after", "before", "during", "while"]
            if q_class == "yn":
                a6ID, a6ans = A6_sentence_selection(question, story)
                ans_sentence = [sentence["sentence"] for sentence in story if sentence["sentenceid"] == a6ID][0]
                answer = "no" if "not" in ans_sentence else "yes"
                # answer = "yes no"
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
                answer = best_answer(question, possible, keyword=hq)

        if "PRP" in nltk.pos_tag([answer.strip().lower()])[0][1]:
            coref_dict = build_coref_dict(story)
            if answer.strip().lower() in coref_dict:
                # print(question["question"])
                # print(answer, end=" ")
                answer = coref_dict[answer.strip().lower()]
                # print(answer)
        ###     End of Your Code         ###
        answerid = "-"
        # print(question["question"])
        # print(answer)
        # print()
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
    # print(keyword)
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
    return ((str(subject), str(headword), str(obj)))

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
    doc = get_story_nlp(story)
    for word in doc:
        if word.lemma_ == token.lemma_:
            return word

def head_of_question(question, story):
    the_story_set = list()
    doc = nlp(question["question"])
    text = ""
    sentences = []
    for sent in story:
        sentences.append(sent["sentence"])
        text += sent["sentence"] + " "
    sdoc = nlp(text)

    for qtoken in doc:
        for stoken in sdoc:
            if qtoken.lemma_.lower() == stoken.lemma_.lower() and qtoken not in the_story_set:
                the_story_set.append(qtoken)
    the_ss_l= [token.lemma_.lower() for token in the_story_set]
    if (len(the_story_set) == 0):
        return doc[0]

    if question_class(question) in ["who"]:
        # print("QUESTION:", question["question"])
        # print(the_story_set)
        tok = doc[0]
        while (tok != tok.head):
            tok = tok.head

        if tok.lemma_ not in the_ss_l or tok.text in stop:
            chunks = [chunk for chunk in doc.noun_chunks]
            if len(chunks) > 1:
                for chunk in reversed(chunks):
                    if chunk.root.text not in stop:
                        if chunk.root.text.lower() == chunk.root.text and chunk.root.text in the_ss_l:
                            return chunk.root
            if "ADJ" in [token.pos_ for token in the_story_set]:
                for token in doc:
                    if token.pos_ == "ADJ" and token in the_story_set:
                        # print(token.text)
                        return token

            else:
                maxtoken = the_story_set[0]
                for token in the_story_set:
                    if len(token.text) >= len(maxtoken.text):
                        maxtoken = token
                # print(maxtoken)
                return maxtoken
        else:
            # print(tok.text)
            return tok

    elif question_class(question) in ["what"]:
        # print("QUESTION:", question["question"])
        # print(the_story_set)
        tok = doc[0]
        while (tok != tok.head):
            tok = tok.head
        if tok.lemma_ not in the_ss_l or tok.text in stop:
            verbs = [token for token in the_story_set if token.pos_ == "VERB"]
            nouns = [token for token in the_story_set if token.pos_ == "NOUN"]
            if len(verbs)>0:
                # print("Chose a verb")
                return verbs[len(verbs)-1]
            if len(nouns)>0:
                # print("Chose a noun")
                return nouns[len(nouns)-1]
            if "ADJ" in [token.pos_ for token in the_story_set]:
                for token in doc:
                    if token.pos_ == "ADJ" and token in the_story_set:
                        # print(token.text)
                        return token
            else:
                maxtoken = the_story_set[0]
                for token in the_story_set:
                    if len(token.text) >= len(maxtoken.text):
                        maxtoken = token
                # print(maxtoken)
                return maxtoken
        else:
            # print(tok.text)
            return tok

    elif question_class(question) in ["when"]:
        # print("QUESTION:", question["question"])
        # print(the_story_set)
        tok = doc[0]
        while (tok != tok.head):
            tok = tok.head

        if tok.lemma_ not in the_ss_l or tok.text in stop:
            chunks = [chunk for chunk in doc.noun_chunks]
            if len(chunks) > 1:
                for chunk in reversed(chunks):
                    if chunk.root.text not in stop:
                        if chunk.root.text.lower() == chunk.root.text and chunk.root.text in the_ss_l:
                            return chunk.root
            if "ADJ" in [token.pos_ for token in the_story_set]:
                for token in doc:
                    if token.pos_ == "ADJ" and token in the_story_set:
                        # print(token.text)
                        return token

            else:
                maxtoken = the_story_set[0]
                for token in the_story_set:
                    if len(token.text) >= len(maxtoken.text):
                        maxtoken = token
                # print(maxtoken)
                return maxtoken
        else:
            # print(tok.text)
            return tok

    elif question_class(question) in ["why", "how"]:
        # print("QUESTION:", question["question"])
        # print(the_story_set)
        tok = doc[0]
        while (tok != tok.head):
            tok = tok.head

        if tok.lemma_ not in the_ss_l or tok.text in stop:
            chunks = [chunk for chunk in doc.noun_chunks]
            if len(chunks) > 1:
                for chunk in reversed(chunks):
                    if chunk.root.text not in stop:
                        if chunk.root.text.lower() == chunk.root.text and chunk.root.text in the_ss_l:
                            return chunk.root
            if "ADJ" in [token.pos_ for token in the_story_set]:
                for token in doc:
                    if token.pos_ == "ADJ" and token in the_story_set:
                        # print(token.text)
                        return token

            else:
                maxtoken = the_story_set[0]
                for token in the_story_set:
                    if len(token.text) >= len(maxtoken.text):
                        maxtoken = token
                # print(maxtoken)
                return maxtoken
        else:
            # print(tok.text)
            return tok

    elif question_class(question) in ["where"]:
        # print("QUESTION:", question["question"])
        # print(the_story_set)
        tok = doc[0]
        while (tok != tok.head):
            tok = tok.head

        if tok.lemma_ not in the_ss_l or tok.text in stop:
            chunks = [chunk for chunk in doc.noun_chunks]
            if len(chunks) > 1:
                for chunk in reversed(chunks):
                    if chunk.root.text not in stop:
                        if chunk.root.text.lower() == chunk.root.text and chunk.root.text in the_ss_l:
                            return chunk.root
            if "ADJ" in [token.pos_ for token in the_story_set]:
                for token in doc:
                    if token.pos_ == "ADJ" and token in the_story_set:
                        # print(token.text)
                        return token

            else:
                maxtoken = the_story_set[0]
                for token in the_story_set:
                    if len(token.text) >= len(maxtoken.text):
                        maxtoken = token
                # print(maxtoken)
                return maxtoken
        else:
            # print(tok.text)
            return tok

    elif question_class(question) in ["yn"]:
        # print("QUESTION:", question["question"])
        # print(the_story_set)
        tok = doc[0]
        while (tok != tok.head):
            tok = tok.head

        if tok.lemma_ not in the_ss_l or tok.text in stop:
            chunks = [chunk for chunk in doc.noun_chunks]
            if len(chunks) > 1:
                for chunk in reversed(chunks):
                    if chunk.root.text not in stop:
                        if chunk.root.text.lower() == chunk.root.text and chunk.root.text in the_ss_l:
                            return chunk.root
            if "ADJ" in [token.pos_ for token in the_story_set]:
                for token in doc:
                    if token.pos_ == "ADJ" and token in the_story_set:
                        # print(token.text)
                        return token

            else:
                maxtoken = the_story_set[0]
                for token in the_story_set:
                    if len(token.text) >= len(maxtoken.text):
                        maxtoken = token
                # print(maxtoken)
                return maxtoken
        else:
            # print(tok.text)
            return tok
SAVED_COREF = ("a", dict())
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
        for i in range(0, len(localstory)):
            # print(story[i]["sentence"])
            localstory[i]["sentence"] = nltk.word_tokenize(
                localstory[i]["sentence"])
            # print(localstory[i]["sentence"])

        for key in story_coref:
            # print("new antecedent")
            antecedent = story_coref[key][0]["text"]
            for z in range(1, len(story_coref[key])):
                sentind = story_coref[key][z]["sentNum"]
                example_text = story_coref[key][z]["text"]
                antecedent_tokens = nltk.word_tokenize(antecedent)
                #print("length of local story:", len(localstory), "title:", localstory[0]["storytitle"], "sentind:", sentind)
                example_tokens = nltk.word_tokenize(example_text)
                #print("example_tokens:", example_tokens)
                if sentind>len(localstory):
                    sentind = sentind-1
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
            localstory[i]["sentence"] = " ".join(localstory[i]["sentence"])
        SAVED_COREF = (story[0]["storyid"], localstory)
        return localstory

def build_coref_dict(story):
    coref_dict = {}
    for sentence in story:
        coreferences = sentence["coref"]
        for chain_key in coreferences:
            chain = coreferences[chain_key]
            if len(chain)>1:
                for reference in chain[1:]:
                    # print(chain[0]["text"])
                    coref_dict.update({reference["text"].lower(): chain[0]["text"]})
    return coref_dict



def get_story_nlp(story):
    text = ""
    sentences = []
    for sent in story:
        sentences.append(sent["sentence"])
        text += sent["sentence"] + " "
    return nlp(text)

def find_in_story2(doc, token):
    best_choice = doc[0]
    matches=[]
    for word in doc:
        if word.lemma_.lower() == token.lemma_.lower():
            matches.append(word)
    if len(matches)>0:
        best_choice=matches[0]
    for word in doc:
        if word.text.lower() == token.text.lower():
            best_choice = word
            break
    return best_choice

def check_if_pronoun_and_resolve(answer, story, question):
    extracted_string= ""
    #if len([value for value in ["some", "My", "Me", "my" "me", "I", "He", "She", "They", "he", "she",
    #   "they", "hers", "her", "him", "his","Hers" "Her", "Him", "His"] if value in nltk.word_tokenize(answer)]
    #       )!=0:
    if answer in ["it", "It", "some", "My", "Me", "my" "me", "I", "He", "She", "They", "he", "she","they", "hers", "her", "him", "his","Hers" "Her", "Him", "His"]:
        if question_class(question)=="who":
            extracted_string=extract_who_answer(coreference_story(story), question, 1)
        if question_class(question) == "what":
            extracted_string=extract_what_answer(coreference_story(story), question, 1)

    else:
        extracted_string = answer
    return extracted_string

def extract_what_answer(story, question, recur_count):
    token = head_of_question(question, story)
    doc = get_story_nlp(story)
    docq = nlp(question["question"])
    if token == docq[0]:
        ent = [e for e in doc.ents][0]
        token = [t for t in doc if t.text == ent.text][0]
    if token == docq[0]:
        token = doc[0]
    best_choice = find_in_story2(doc, token)
    # print("best_choice:", best_choice)
    if docq[1].text not in ["did"]:
        # print("not a did")
        for dep in ["nsubj", "nsubjpass", "aux", "dsubj"]:
            for child in best_choice.children:
                # print((child.text, child.dep_))
                if child.dep_ == dep:
                    # print("Chosen dep_:", child.dep_)
                    answer_string = " ".join([token.text for token in list(child.subtree)])
                    if recur_count == 0:
                        answer_string = check_if_pronoun_and_resolve(answer_string, story, question)
                    # print("nsubj subtree string:", answer_string)
                    return answer_string
    elif len([noun for noun in docq if noun.pos_ in ['NOUN','PROPN']])>=2:
        # print("Two nouns in the question")
        while(best_choice != best_choice.head):
            best_choice = best_choice.head
        # print("HEAD:", best_choice.text)
        for i in range(0,len(list(best_choice.rights))):
            right =[token for token in best_choice.rights][i]
            # print("went right", right.text)
            for dep in ["dobj", "pobj"]:
                for child in right.children:
                    # print((child.text, child.dep_))
                    if child.dep_ == dep:
                        # print("Chosen dep_:", child.dep_)
                        answer_string = " ".join([token.text for token in list(child.subtree)])
                        if recur_count == 0:
                            answer_string = check_if_pronoun_and_resolve(answer_string, story, question)
                        # print("nsubj subtree string:", answer_string)
                        return answer_string
    else:
        while (best_choice != best_choice.head):
            best_choice = best_choice.head
            # print("HEAD:", best_choice.text)
        for dep in ["dobj", "pobj", "relcl", "xcomp", "ccomp", "conj", "advcl", "prep"]:
            for child in best_choice.children:
                # print((child.text, child.dep_))
                if child.dep_ == dep:
                    # print("Chosen dep_:", child.dep_)
                    answer_string = " ".join([token.text for token in list(child.subtree)])
                    if recur_count == 0:
                        answer_string = check_if_pronoun_and_resolve(answer_string, story, question)
                    # print("nsubj subtree string:", answer_string)
                    return answer_string


    chunks = [chunk.text for chunk in doc.noun_chunks if chunk.text not in [t.text for t in docq]]
    if len(chunks) > 0:
        chunk1 = chunks[0]
        # print("Noun returned", chunk1)
        return chunk1
    else:
        # print("nothing")
        return " "

def extract_who_answer(story, question, recur_count):

    token = head_of_question(question, story)
    doc = get_story_nlp(story)
    docq = nlp(question["question"])
    #IF THERE WERE NO WORDS FROM THE WHO-QUESTION IN THE STORY I RETURNED THE FIRST TOKEN OF THE QUESTION and
    if token == docq[0]:
        ent = [e for e in doc.ents][0]
        token = [t for t in doc if t.text == ent.text][0]
    if token == docq[0]:
        token= doc[0]
    best_choice= find_in_story2(doc, token)
    # print("best_choice:", best_choice)
    question_people = [e.text for e in docq.ents if
                       (e.label_ == 'PERSON') or (e.label_ == 'ORG') or (e.label_ == 'GPE')]
    # print(question_people)
    people = [e.text for e in doc.ents if (e.label_ == 'PERSON') or (e.label_ == 'ORG') or (e.label_ == 'GPE')]
    # print(people)
    other_people = [person for person in people if person not in [token.text for token in docq]]

    if token.i == 1:
        # print("is second word")
        for child in best_choice.children:
            if child.dep_ in ["nsubj", "nsubjpass", "aux"]:
                if not (child.dep_ =="aux" and child.i>best_choice.i):
                    answer_string = " ".join([token.text for token in list(child.subtree)])
                    if recur_count == 0:
                        answer_string = check_if_pronoun_and_resolve(answer_string,story,question)
                    # print("nsubj subtree string:", answer_string)
                    return answer_string
        if best_choice.dep_ == "conj":
            for child in best_choice.head.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    answer_string = " ".join([token.text for token in list(child.subtree)])
                    if recur_count==0:
                        answer_string=check_if_pronoun_and_resolve(answer_string,story, question)
                    # print("nsubj subtree string:", answer_string)
                    return answer_string
        if best_choice.dep_ == "xcomp":
            for child in best_choice.head.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    answer_string = " ".join([token.text for token in list(child.subtree)])
                    if recur_count==0:
                        answer_string=check_if_pronoun_and_resolve(answer_string,story,question)
                    # print("nsubj subtree string:", answer_string)
                    return answer_string

    else:
        # print("is not second word")
        while (best_choice != best_choice.head):
            best_choice= best_choice.head
        if len(question_people)!=0:
            if len(other_people) > 0:
                other_person_string = other_people[0]
                # print("Other person returned", other_person_string)
                return other_person_string
        else:
            for child in best_choice.children:
                if child.dep_ in ["nsubj", "nsubjpass", "aux"]:
                    answer_string = " ".join([token.text for token in list(child.subtree)])
                    if recur_count == 0:
                        answer_string = check_if_pronoun_and_resolve(answer_string, story, question)
                    # print("nsubj subtree string:", answer_string)
                    return answer_string
        #else:
        #  for child in best_choice.children:
        #     if child.dep_ in ["nobj", "pobj"]:
        #         answer_string = " ".join([token.text for token in list(child.subtree)])
        #         if recur_count == 0:
        #             answer_string = check_if_pronoun_and_resolve(answer_string, story, question)
        #         print("nobj/pobj subtree string:", answer_string)
        #          return answer_string




        chunks = [chunk.text for chunk in doc.noun_chunks if chunk.text not in [t.text for t in docq]]
        if len(chunks) > 0:
            chunk1 = chunks[0]
            # print("Noun returned", chunk1)
            return chunk1
        else:
            # print("nothing")
            return " "

    return "a"

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
