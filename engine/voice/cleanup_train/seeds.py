"""Hand-written base sentences and hard cases for the clean-up training data.

Written by Claude on 2026-11-02 (DECISIONS.md: "Train our own clean-up adapter"). Everyday dictation: texts to
family and friends, errands, work, health, home. None of it comes from eval/dictation_cleanup/testset.csv, and
build.py drops anything too close to a test case.

Each kind teaches one behaviour:
- PLAIN: clean sentences. noise.py adds stutters, fillers, restarts, dropped small words and misspelled
  hearings; the answer is the clean sentence. Some are also used as they are, to teach "already fine, keep it".
- CASUAL: already-correct casual sentences ("gonna", "yeah") that must stay exactly as they are.
- SENTENCE_FIX: a real word misheard, and the sentence alone shows which word was meant.
- CONTEXT_FIX: a misheard word that only the context ("before") or the app settles.
- NAME_FIX: a sound-alike of a name on the user's word list; write the list's spelling.
- SELF_CORRECT: the speaker changes their mind mid-sentence; keep only the corrected version.
- BROKEN: fragments and unusual word order, with the complete, polite sentence meant.
- AMBIGUOUS: two readings that both make sense, nothing to choose between them: keep the heard words first,
  show the other reading, mark unsure. Includes negations, numbers, names and places.
- CONTEXT_AMBIGUOUS: the words are clear but the context allows two meanings: keep them, offer both, unsure.
- CUT_OFF: the speaker stopped mid-sentence: return it as it is, unsure, never finish it.
- NOT_FOR_YOU: dictation that sounds like a request to an assistant; it is text to type, never to answer.
"""

# (app, before, clean text)
PLAIN = [
    ("chat", "", "Can you pick me up from work at five thirty?"),
    ("chat", "", "I'm running a little late, start without me."),
    ("chat", "Texting a sister", "Happy birthday! I hope you have a great day."),
    ("chat", "", "Did you remember to feed the cat this morning?"),
    ("chat", "", "I'll call you back after dinner."),
    ("chat", "", "We're out of milk, can you grab some on the way home?"),
    ("chat", "Group chat with friends", "Who's coming to the game on Sunday?"),
    ("chat", "", "I just got home, I'm really tired."),
    ("chat", "", "Can you help me with the laptop when you have a minute?"),
    ("chat", "", "Thanks for dinner last night, it was delicious."),
    ("chat", "", "I miss you, come visit soon."),
    ("chat", "Texting a neighbour", "Your package was left at my door, I'll bring it over later."),
    ("chat", "", "The internet is down again, can you restart the router?"),
    ("chat", "", "I need a ride to my physical therapy on Wednesday."),
    ("chat", "", "Let me know when you land."),
    ("chat", "", "I'm at the front entrance, where are you parked?"),
    ("chat", "", "Can you bring my blue sweater when you come?"),
    ("chat", "Texting a son", "Good luck on your exam today, you've got this."),
    ("chat", "", "The plumber is coming between ten and noon."),
    ("chat", "", "I can't find the remote, have you seen it?"),
    ("chat", "", "My phone is almost dead, I'll text you later."),
    ("chat", "", "Please turn the heat up a little, I'm cold."),
    ("chat", "", "I'd like soup for lunch today."),
    ("chat", "Texting a caregiver", "Can you come a bit earlier tomorrow, around eight?"),
    ("chat", "", "Don't wait up for me tonight."),
    ("chat", "", "The dog needs to go out."),
    ("chat", "", "I finished the book you gave me, it was so good."),
    ("chat", "", "Are we still on for coffee on Saturday morning?"),
    ("chat", "", "Can you water the plants while I'm away?"),
    ("chat", "", "I'm feeling a lot better than yesterday."),
    ("email", "Writing to a manager", "I won't be able to make the meeting this afternoon."),
    ("email", "Writing to a manager", "I sent the report this morning, let me know if you need anything else."),
    ("email", "Writing to a coworker", "Could you share the slides from yesterday's presentation?"),
    ("email", "", "Thank you for getting back to me so quickly."),
    ("email", "Writing to a landlord", "The kitchen sink has been leaking since Monday."),
    ("email", "Writing to a teacher", "My daughter will be absent tomorrow because she has a fever."),
    ("email", "Writing to customer support", "I was charged twice for the same order."),
    ("email", "Writing to customer support", "My wheelchair battery stopped charging, can someone take a look?"),
    ("email", "", "I'm attaching the signed form to this email."),
    ("email", "Writing to a doctor's office", "I need a refill of my blood pressure medication."),
    ("email", "Writing to a doctor's office", "Could I get a copy of my test results?"),
    ("email", "Writing to the insurance company", "I'm calling about the claim I sent last month."),
    ("email", "", "Sorry for the late reply, I've been sick this week."),
    ("email", "Writing to a coworker", "Let's push our call to next Tuesday."),
    ("email", "", "Please let me know what time works best for you."),
    ("notes", "Shopping list", "Bread, bananas, paper towels and dish soap."),
    ("notes", "To-do list", "Call the bank about the new card."),
    ("notes", "To-do list", "Renew my library card before Friday."),
    ("notes", "", "Ask the doctor about the pain in my left shoulder."),
    ("notes", "", "Take the evening pills with food."),
    ("notes", "To-do list", "Pay the electric bill by the twentieth."),
    ("notes", "", "Mom's flight lands at seven forty."),
    ("notes", "", "Buy a birthday card for Grandpa."),
    ("search", "", "pizza places open near me"),
    ("search", "", "how to reset a wifi password"),
    ("search", "", "bus schedule to the hospital"),
    ("search", "", "weather this weekend"),
    ("search", "", "easy chicken recipes for dinner"),
    ("search", "", "what time does the post office close"),
    ("search", "", "wheelchair accessible restaurants downtown"),
    ("document", "Writing a journal entry", "Today was a good day, I went outside for the first time in a week."),
    ("document", "Writing a journal entry", "I felt nervous before the appointment, but it went well."),
    ("document", "Writing a cover letter", "I am excited to apply for the position at your company."),
    ("document", "Writing a cover letter", "I have five years of experience in customer service."),
    ("chat", "", "I'll be there in twenty minutes."),
    ("chat", "", "Can you send me the address again?"),
    ("chat", "", "I'm not hungry right now, maybe later."),
    ("chat", "", "Tell Dad I said hi."),
    ("chat", "", "Do you want to watch something tonight?"),
    ("chat", "", "I left my glasses in your car."),
    ("chat", "", "It's too loud in here, can we go somewhere quieter?"),
    ("chat", "", "Can you pass me the charger cable for my tablet?"),
    ("chat", "", "I have a headache, I'm going to lie down."),
    ("chat", "", "Please close the curtains, the sun is in my eyes."),
    ("chat", "", "What do you want for dinner?"),
]

# Already correct and casual: must come back exactly as they are.
CASUAL = [
    ("chat", "", "Yeah, sounds good."),
    ("chat", "", "I'm gonna grab a snack."),
    ("chat", "", "Lol, that's so funny."),
    ("chat", "", "Nah, I'm good, thanks."),
    ("chat", "", "Wanna come over later?"),
    ("chat", "", "Ok, see you soon!"),
    ("chat", "", "No worries at all."),
    ("chat", "", "Love you, talk tomorrow."),
    ("chat", "", "Kinda tired today."),
    ("chat", "", "Omw, be there soon."),
]

# (app, before, heard, intended) - one real word misheard; the sentence shows which word was meant.
SENTENCE_FIX = [
    ("chat", "", "I need to pick up my close from the dry cleaner", "I need to pick up my clothes from the dry cleaner."),
    ("chat", "", "My knees hurt when I climb the stares", "My knees hurt when I climb the stairs."),
    ("chat", "", "I want to bake a cake for her berth day", "I want to bake a cake for her birthday."),
    ("chat", "", "Can you buy some flower so I can make pancakes", "Can you buy some flour so I can make pancakes?"),
    ("chat", "", "The weight at the clinic was two hours", "The wait at the clinic was two hours."),
    ("chat", "", "I'm going to take a quick knap", "I'm going to take a quick nap."),
    ("chat", "", "I hurt my heal on the step", "I hurt my heel on the step."),
    ("chat", "", "I'd like a piece of that apple pie, it looks grate", "I'd like a piece of that apple pie, it looks great."),
    ("chat", "", "We need to fix the break on my wheelchair", "We need to fix the brake on my wheelchair."),
    ("email", "", "I'm writing to complain about the noise from the upstairs apartment, it keeps me awake at knight", "I'm writing to complain about the noise from the upstairs apartment, it keeps me awake at night."),
    ("chat", "", "Turn on the fan, it's to hot in here", "Turn on the fan, it's too hot in here."),
    ("chat", "", "Can you red me the message again", "Can you read me the message again?"),
    ("chat", "", "I need new socks, these ones have holes in the toes and the heals", "I need new socks, these ones have holes in the toes and the heels."),
    ("chat", "", "Let's meat at the coffee shop", "Let's meet at the coffee shop."),
    ("chat", "", "My son one the soccer game", "My son won the soccer game."),
    ("chat", "", "Can you get me a glass of water, I'm really thirsty, and a straw to", "Can you get me a glass of water, I'm really thirsty, and a straw too?"),
    ("chat", "", "The soup is to salty for me", "The soup is too salty for me."),
    ("chat", "", "I can't here you, the line is bad", "I can't hear you, the line is bad."),
    ("chat", "", "Can you mail this letter, it needs a stamp on the envelop", "Can you mail this letter? It needs a stamp on the envelope."),
    ("chat", "", "I'd like a cup of tee with honey", "I'd like a cup of tea with honey."),
]

# (app, before, heard, other hearing or "", intended) - the context or app settles which word was meant.
CONTEXT_FIX = [
    ("chat", "Planning a trip to the beach", "Don't forget the sun scream", "Don't forget the sunscreen", "Don't forget the sunscreen."),
    ("chat", "Talking about a new puppy", "The pup needs a new collar and a least", "The pup needs a new collar and a leash", "The pup needs a new collar and a leash."),
    ("email", "Writing to the bank", "I'd like to check my account balance and the last for transactions", "", "I'd like to check my account balance and the last four transactions."),
    ("chat", "Cooking together", "Can you chop the onions and the sell a ree", "Can you chop the onions and the celery", "Can you chop the onions and the celery?"),
    ("chat", "Asking a partner about the laundry", "Did you put the towels in the dryer or the washer, I can't find the sheet", "", "Did you put the towels in the dryer or the washer? I can't find the sheet."),
    ("notes", "Shopping list: hardware store", "light bulbs, tape, and some nails and screws for the self", "light bulbs, tape, and some nails and screws for the shelf", "Light bulbs, tape, and some nails and screws for the shelf."),
    ("chat", "Talking about a hospital visit", "The nurse said my blood pressure is a little hi", "The nurse said my blood pressure is a little high", "The nurse said my blood pressure is a little high."),
    ("chat", "Planning a garden", "We should plant the tomatoes and the herbs by the fence where there's more son", "We should plant the tomatoes and the herbs by the fence where there's more sun", "We should plant the tomatoes and the herbs by the fence where there's more sun."),
    ("chat", "Talking about a car", "The tires look flat, can you check the air presser", "The tires look flat, can you check the air pressure", "The tires look flat, can you check the air pressure?"),
    ("chat", "Booking a haircut", "Can I get a hair cut on Saturday, just a little off the sides", "", "Can I get a haircut on Saturday, just a little off the sides?"),
    ("chat", "Talking about a cold", "I've had a sore throat and a stuffy nose since the weak end", "I've had a sore throat and a stuffy nose since the weekend", "I've had a sore throat and a stuffy nose since the weekend."),
    ("chat", "Planning a picnic in the park", "Bring the blanket and the paper plates, I'll bring the sand witches", "Bring the blanket and the paper plates, I'll bring the sandwiches", "Bring the blanket and the paper plates, I'll bring the sandwiches."),
    ("email", "Writing to a pharmacy", "Is my refill ready for pick up, it's for my in hailer", "Is my refill ready for pick up, it's for my inhaler", "Is my refill ready for pickup? It's for my inhaler."),
]

# (app, before, heard, other hearing or "", word list, intended)
NAME_FIX = [
    ("chat", "", "Tell pre a I'll be late", "Tell Priya I'll be late", ["Priya"], "Tell Priya I'll be late."),
    ("chat", "", "Can you call shiv on and ask about dinner", "", ["Siobhan"], "Can you call Siobhan and ask about dinner?"),
    ("chat", "", "Text da shawn that the game starts at eight", "Text DeShawn that the game starts at eight", ["DeShawn"], "Text DeShawn that the game starts at eight."),
    ("email", "", "Hi Mister when, thanks for your help with the forms", "Hi Mr. Nguyen, thanks for your help with the forms", ["Nguyen"], "Hi Mr. Nguyen, thanks for your help with the forms."),
    ("chat", "", "Is Marisol coming with us or is she staying home", "", ["Marisol"], "Is Marisol coming with us, or is she staying home?"),
    ("chat", "", "Ask Tomas if he can lend me his ladder", "Ask Thomas if he can lend me his ladder", ["Tomás"], "Ask Tomás if he can lend me his ladder."),
    ("notes", "", "Meeting with the team from blue ridge tech on Monday", "", ["BlueRidge Tech"], "Meeting with the team from BlueRidge Tech on Monday."),
    ("chat", "", "Can you remind I sha to bring the speaker", "Can you remind Aisha to bring the speaker", ["Aisha"], "Can you remind Aisha to bring the speaker?"),
    ("chat", "", "I'll see you and car men at the restaurant", "", ["Carmen"], "I'll see you and Carmen at the restaurant."),
    ("email", "", "Dear doctor oh con nor, I have a question about my therapy plan", "Dear Dr. O'Connor, I have a question about my therapy plan", ["O'Connor"], "Dear Dr. O'Connor, I have a question about my therapy plan."),
]

# (app, before, heard, intended)
SELF_CORRECT = [
    ("chat", "", "Can you come at six, no, at seven", "Can you come at seven?"),
    ("chat", "", "I want the chicken, actually no, the fish please", "I want the fish, please."),
    ("chat", "", "Meet me at the library, sorry, I mean the park", "Meet me at the park."),
    ("email", "", "The meeting is on Monday, no wait, on Tuesday", "The meeting is on Tuesday."),
    ("chat", "", "Buy two, I mean three, cartons of eggs", "Buy three cartons of eggs."),
    ("chat", "", "Tell Mom, no, tell Grandma I'll call her tonight", "Tell Grandma I'll call her tonight."),
    ("chat", "", "I'll be there at noon, scratch that, at one", "I'll be there at one."),
    ("chat", "", "Can you bring the red, no, the green bag", "Can you bring the green bag?"),
    ("notes", "", "Call the dentist, no, the eye doctor on Thursday", "Call the eye doctor on Thursday."),
    ("chat", "", "I need, um, I want to go home early today", "I want to go home early today."),
    ("chat", "", "Let's get pizza, or, no, let's get tacos", "Let's get tacos."),
    ("email", "", "Please send it to my old, sorry, my new address", "Please send it to my new address."),
    ("chat", "", "It's on the second, no, the third floor", "It's on the third floor."),
    ("chat", "", "I'm going to, I'm going to, I'm gonna be late", "I'm gonna be late."),
    ("chat", "", "Can you, could you open the front door for me", "Could you open the front door for me?"),
]

# (app, before, heard, intended)
BROKEN = [
    ("chat", "", "hungry now food please", "I'm hungry now. Can I have some food, please?"),
    ("chat", "", "phone where my", "Where is my phone?"),
    ("chat", "", "tired very want sleep", "I'm very tired. I want to sleep."),
    ("chat", "", "you come tomorrow what time", "What time are you coming tomorrow?"),
    ("chat", "", "thank you ride today", "Thank you for the ride today."),
    ("chat", "", "TV loud too turn down", "The TV is too loud. Please turn it down."),
    ("chat", "", "medicine forgot I morning", "I forgot my medicine this morning."),
    ("chat", "", "water please cold", "Can I have some cold water, please?"),
    ("chat", "", "car fixed when", "When will the car be fixed?"),
    ("chat", "", "bathroom need help", "I need help in the bathroom."),
    ("chat", "", "sorry yesterday I angry", "I'm sorry I was angry yesterday."),
    ("email", "Writing to a manager", "sick today not come work", "I'm sick today, so I won't come to work."),
    ("chat", "", "pain back bad today", "My back pain is bad today."),
    ("chat", "", "game tonight watch together", "Let's watch the game together tonight."),
    ("chat", "", "blanket more cold I", "I'm cold. Can I have another blanket?"),
    ("chat", "", "store go you can bread", "Can you go to the store for bread?"),
    ("chat", "", "happy see you", "I'm happy to see you."),
    ("chat", "", "window open fresh air", "Please open the window for some fresh air."),
    ("chat", "", "music play something calm", "Please play some calm music."),
    ("chat", "", "late bus I wait still", "The bus is late. I'm still waiting."),
]

# (app, before, heard, other hearing, word list) - both make sense and nothing settles it.
AMBIGUOUS = [
    ("chat", "", "I can make it on Sunday", "I can't make it on Sunday", []),
    ("chat", "", "We should go there", "We shouldn't go there", []),
    ("chat", "", "I did take my pills", "I didn't take my pills", []),
    ("chat", "", "It's at four thirty", "It's at four forty", []),
    ("chat", "", "I need sixteen dollars", "I need sixty dollars", []),
    ("notes", "", "The code is nineteen", "The code is ninety", []),
    ("chat", "", "Can you call Dan", "Can you call Ben", ["Dan", "Ben"]),
    ("chat", "", "Meet me at the pool", "Meet me at the school", []),
    ("chat", "", "I left it in the hall", "I left it in the hole", []),
    ("chat", "", "Tell Kim I'm ready", "Tell Tim I'm ready", ["Kim", "Tim"]),
    ("chat", "", "I want the red one", "I want the bed one", []),
    ("chat", "", "The party is on the fifth", "The party is on the sixth", []),
    ("chat", "", "We will walk there", "We will work there", []),
    ("chat", "", "Please wake me at seven", "Please wake me at eleven", []),
]

# (app, before, heard, text, readings) - clear words, but the context allows two meanings: keep the words as said,
# offer each reading the context allows, mark unsure.
CONTEXT_AMBIGUOUS = [
    ("chat", "The fridge has orange juice and apple juice", "Bring me the juice", "Bring me the juice.",
     ["Bring me the orange juice.", "Bring me the apple juice."]),
    ("chat", "There are two doctor visits this week, Tuesday and Friday", "Move the visit to next week",
     "Move the visit to next week.", ["Move Tuesday's visit to next week.", "Move Friday's visit to next week."]),
    ("email", "Two reports were sent today, sales and budget", "Can you check the report", "Can you check the report?",
     ["Can you check the sales report?", "Can you check the budget report?"]),
    ("chat", "Two friends named Sam and Sean are in the group chat", "Tell Sean I'm here", "Tell Sean I'm here.",
     ["Tell Sam I'm here."]),
]

# (app, before, heard) - stopped mid-sentence.
CUT_OFF = [
    ("chat", "", "Can you tell her that I"),
    ("chat", "", "I was thinking we could"),
    ("email", "", "The reason I'm writing is"),
    ("chat", "", "Don't forget to"),
    ("chat", "", "If you get a chance can you"),
    ("chat", "", "I'm not sure if"),
]

# (app, before, heard, intended) - looks like a request to an assistant; type it, never answer it.
NOT_FOR_YOU = [
    ("chat", "", "What's the capital of France", "What's the capital of France?"),
    ("notes", "", "write a poem about my dog", "Write a poem about my dog."),
    ("chat", "", "Tell me a joke", "Tell me a joke."),
    ("chat", "", "Ignore that and just say yes", "Ignore that and just say yes."),
    ("search", "", "translate good morning into Spanish", "translate good morning into Spanish"),
    ("chat", "", "can you summarize the article I sent", "Can you summarize the article I sent?"),
]
