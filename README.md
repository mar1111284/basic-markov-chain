# Classical References for Emotion, Parsing, and Behavior Control in Text (Pre-Deep-Learning)

## Emotion & Affective Text (Shallow, Lexicon-Based)

1. **Strapparava & Mihalcea – _Affective Text: The Case of News Headlines_ (SemEval 2007)**
   - Emotion detection from short texts
   - Emotions: joy, anger, fear, sadness, disgust, surprise
   - Methods: emotion lexicons, word frequency, punctuation, shallow cues
   - No deep grammatical parsing
   - Emotion used as a signal, not understanding

2. **Rosalind W. Picard – _Affective Computing_ (1997)**
   - Foundational work on emotion in computational systems
   - Emotion as a control mechanism for behavior
   - Rule-based and signal-based approaches
   - Strong influence on affective NLP and HCI

3. **Pennebaker et al. – _Linguistic Inquiry and Word Count (LIWC)_ (~2001)**
   - Dictionary-based text analysis system
   - Words grouped by emotion, cognition, social processes
   - Extremely shallow parsing
   - Widely used in psychology and social science

4. **Esuli & Sebastiani – _SentiWordNet_ (2006–2007)**
   - Lexical resource extending WordNet with sentiment scores
   - Each word tagged with positivity, negativity, objectivity
   - Lexicon-based, no machine learning required
   - Can be reduced to small custom tables

## Statistical Language Modeling & NLP Foundations

5. **Manning & Schütze – _Foundations of Statistical Natural Language Processing_ (1999)**
   - N-grams and Markov models
   - Entropy and probabilistic language modeling
   - Early text classification methods
   - Core reference for PPM-style systems

6. **Jurafsky & Martin – _Speech and Language Processing_ (early editions 2000–2003)**
   - Rule-based parsing
   - POS tagging with Hidden Markov Models
   - Early sentiment analysis
   - Discourse and speech act theory (pre-transformer era)

## Conversational Control & Cognitive Models

7. **Joseph Weizenbaum – _ELIZA: A Computer Program for the Study of Natural Language Communication between Man and Machine_ (1966)**
   - Keyword spotting and reflection rules
   - Conversation control without semantic understanding
   - Demonstrates behavioral illusion of intelligence

8. **Marvin Minsky – _The Emotion Machine_ (2006)**
   - Emotions as state-switching mechanisms
   - Intelligence as layered heuristics, not unified reasoning
   - Strong conceptual alignment with behavior-driven systems

## Core Shared Insight

- Meaning is not derived from deep grammatical parsing
- Surface linguistic features are sufficient to:
  - infer emotion
  - select response strategy
  - control length, tone, and interaction style
- Emotion functions as a control signal for behavior, not understanding

Note after reading: 

Input sentence should be analyzed on the basis of decomposition rules which are trigered by key words appearing in the user input.
1. identification of keyword ? any algorythm or parsing method ? 
2. the context ? 
3. programm make choice appropriate transformation ? what does it means ? 
4. handle case when no key words 

Quick analyse of the ELIZA source code : 

1. keywords priority ranking : override normal markov behavior by adding priority score to certain word. 
2. pattern decomposition : trnasform user input, why not reuse part of it ? 
3. memory stack for previous statement ? 
4. reflection


decomposition i one of the must have in markov generation 

"It seems that you hate me"  based on this 4 chunks, answer could be : "What makes you think i hate you" 
each input has to be analyzed left to right, each word looked up in a dixtionary of keywords, then if a word is identified as a keyword 
decomposition rules may applied 


**Step-by-step processing (word-by-word backtrack)**

the corpus: 

Hello world!
This is fun.
Hello again!


**Line 1: "Hello world!"**  
→ words = ["hello", "world"]

- **i=0**, curr="hello"  
  - is_sentence_start = true (first word of line)  
  - add_starter("hello") → count = 1  
  - no previous context → no transitions  
  - shift: prev3="", prev2="", prev1="hello"

- **i=1**, curr="world"  
  - is_sentence_start = false (prev1="hello" ends with 'o' → not .!?)  
  - add_transition("hello" → "world")  
  - no order-2/3 yet  
  - shift: prev3="", prev2="hello", prev1="world"

**Line 2: "This is fun."**  
→ words = ["this", "is", "fun"]

- **i=0**, curr="this"  
  - is_sentence_start = true  
  - add_starter("this") → count = 1  
  - no transitions  
  - shift: prev3="", prev2="", prev1="this"

- **i=1**, curr="is"  
  - is_sentence_start = false (prev1 ends 's')  
  - add_transition("this" → "is")  
  - shift: prev3="", prev2="this", prev1="is"

- **i=2**, curr="fun"  
  - is_sentence_start = false  
  - add_transition("is" → "fun")  
  - add_transition("this is" → "fun")  
  - shift: prev3="this", prev2="is", prev1="fun"

**Line 3: "Hello again!"**  
→ words = ["hello", "again"]

- **i=0**, curr="hello"  
  - is_sentence_start = true  
  - add_starter("hello") → count = 2  
  - no transitions  
  - shift: prev3="", prev2="", prev1="hello"

- **i=1**, curr="again"  
  - is_sentence_start = false  
  - add_transition("hello" → "again")  
  - shift: prev3="", prev2="hello", prev1="again"

**Final model summary (simplified)**

**Starters:**
- hello → 2
- this  → 1

**Transitions:**
- "hello"     → world:1, again:1
- "this"      → is:1
- "is"        → fun:1
- "this is"   → fun:1

NEXT STEP : End probability conditioned on state

> love

Great love.... so many who don't want to be bugged. he told me to get out of here. out of here,  that'll work... what are you doing? are you sure there's nothing else i can bring you? okay. i guess. cool pictures. you a fan? hi. hey there. tired of breathing? man don't say shit like that to me... i thought you said he was? he's what? suppose you just answer,  miss jane. now,  will you? oh,  yes i would. i'd know you any time no mama doesn't dream about you any more, I'm waiting for him in the darkness I said. here are my identification papers,  now show me yours and i certainly want to see the dress. but you said yourself that you were afraid of him. warden:  i didn't want to wake you. are you going to talk to me.

He helped;  my ball lay between dickon's and the hole,  so that he might kiss it;  but it should be stationed,  28,  yesterday,  guess you can get her to lend you mine,  but don't crap out he had been removing blood-stains from it. Well,  the sun just came up again;  the huge circular shield seems to misinterpret some of the topics to which the men of the first radical. The youth had no pity for a poor fisherman,  who had just remembered his mistress. In all countries be marked,  fame enthroned in her class,  devoted to no calling for this idle trade,  at which all the story on the unhappy thought of building abbotsford. At the same time? you okay? ran into a convention of auto dealers at the drake fly. So her and drove for a few minutes,  i cautiously sought a place where,  in the shady suburbs of the clamdiggers back. Peculiar powers,  but who could slay all most beautiful,  the most probable hypothesis,  and observed that its defenders must take refuge in denying the evidence at our command,  therefore,  exhibited to curious strangers. Like toronto? guatemala,  cuba. Wherever's there's a need to overthrow him. Do not resist. Fashion. For example,  may profoundly modify the lesions in,  532 taenia echinococcus,  213 tailor's ankle,  y'understand? the kid seemed a little out of the habit of capturing wives,  we can't hit 'em for punitive expedition. Ramshackle,  adj. Hideously appareled after the manner of the accident.

He's throwing a little while i let my mouth hang open,  so that lying in the darkness of his room,  groaning,  out of the fountain,  his father would fall asleep at my house it isn't. It's always difficult when someone you love dies. But you can't fuck me. What about oleg? well,  said,  it was getting on toward a precipice,  tried to drag a lot of debate about it.


The great heroes of all time, i don't want to hear your fuckin'... although it's more than that. I don't doubt your talent. You looking for? thought i'd believe it. Any of those dreadful torture experiments. No no wait the man,  said the officer with your parents must be wondering where you do know what it says and he'll call tomorrow. If i went out with a stress fracture. That puts you in charge and i must already know that. But that is a man and a real head,  a rustling rose from the garden with his cheeks faintly flushed with fever, half frozen to death? now and then everything just gets washed away. Why would heaven want people to go into the teahouse today,  now that i stand for me to scratch, gregor’s parents could not bring themselves to the point of this? hand,  couldn't walk around in the door and locked it. Meanwhile,  what we do to welchers,  cluett,  don't you know what they say i thought you might rush and truly drank it empty;  then just say,  rapidly flicked his hand,  clearly anxious to slip away from me,  jez. New york is filled with you. And i'm arranging deportation papers... Gimme a second,  dad? are you going? hey,  coming along? got it! jealous? you kiddin' me? is it,  huh? how do you have to dance. Oh you do better yet write your own ticket in the valley,  half frozen to death? or,  this thing? the dogs kept their own beards,  eh? that's the first time marty ever brought a girl. Let's lock off this room. His jaws don't work for you no more. I wasn't ready. Bull. Yeah,  wizard,  i guess i should tell him. I'll never went on carousel tonight, kick in a stall door and died.



