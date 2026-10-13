import { RNG } from "../engine/rng";

/**
 * The large text corpus used by the word-level language model dataset.
 *
 * It is generated deterministically from hand-written sentence banks plus
 * combinatorial templates, so the app ships a genuinely large and varied
 * training text without bundling a multi-megabyte asset. Everything is derived
 * at runtime from a seed, which means the same seed always yields the same
 * corpus, and therefore the same tokenizer.
 *
 * Style: lowercase, ascii, no apostrophes, one sentence per line.
 */

export const LM2_SEQ_LEN = 16;
export const LM2_VOCAB_CAP = 3000;
export const LM2_MAX_WINDOWS = 80000;

const ADJ = [
  "small", "quiet", "bright", "narrow", "ancient", "careful", "curious", "gentle", "patient", "clever",
  "warm", "cold", "quiet", "swift", "brave", "calm", "eager", "honest", "lively", "stubborn",
  "muddy", "golden", "silent", "restless", "frozen", "dusty", "polished", "crooked", "generous", "curious",
];

const NOUN = [
  "river", "garden", "lantern", "notebook", "mountain", "kettle", "bridge", "forest", "window", "library",
  "harbour", "meadow", "compass", "engine", "violin", "blanket", "orchard", "tunnel", "beacon", "workshop",
  "puzzle", "ladder", "market", "harvest", "winter", "summer", "morning", "evening", "journey", "question",
];

const ANIMAL = [
  "fox", "owl", "heron", "otter", "badger", "sparrow", "salmon", "wolf", "hare", "moose",
  "penguin", "camel", "turtle", "raven", "deer", "bee", "spider", "whale", "gecko", "lynx",
];

const VERB = [
  "wandered", "listened", "climbed", "measured", "carried", "followed", "sketched", "counted", "polished", "repaired",
  "planted", "gathered", "studied", "crossed", "mapped", "guarded", "translated", "balanced", "recorded", "observed",
];

const PLACE = [
  "the old harbour", "a quiet library", "the northern ridge", "a small kitchen", "the city archive",
  "a mountain trail", "the botanical garden", "an empty theatre", "the riverside market", "a frozen lake",
  "the workshop bench", "a hillside village", "the museum steps", "a narrow alley", "the coastal road",
];

const NAME = [
  "ada", "ben", "clara", "dev", "elif", "farid", "grace", "hiro", "ines", "jonas",
  "kira", "liam", "mira", "noah", "olga", "pedro", "quinn", "rosa", "samir", "tessa",
  "umar", "vera", "will", "xena", "yusuf", "zara", "amara", "bruno", "cleo", "dario",
];

const FOOD_WORDS = [
  "bread", "soup", "rice", "lentils", "apples", "cheese", "honey", "olives", "peppers", "potatoes",
  "carrots", "onions", "garlic", "ginger", "lemons", "berries", "almonds", "oats", "yogurt", "fish",
];

const COLOUR = [
  "red", "orange", "yellow", "green", "blue", "indigo", "violet", "amber", "teal", "crimson",
  "silver", "copper", "ivory", "ochre", "magenta", "olive", "navy", "coral", "jade", "rust",
];

const JOB = [
  "baker", "carpenter", "nurse", "teacher", "engineer", "botanist", "archivist", "sailor", "potter", "surveyor",
  "translator", "miner", "weaver", "chemist", "librarian", "mechanic", "gardener", "astronomer", "printer", "shepherd",
];

const MONTH = [
  "january", "february", "march", "april", "may", "june",
  "july", "august", "september", "october", "november", "december",
];

const DAY = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];

/* ------------------------------- hand written banks ------------------------------- */

const GREETINGS = [
  "hello there, how are you today",
  "good morning, i hope you slept well",
  "good afternoon, what can i do for you",
  "good evening and welcome back to the lab",
  "hey friend, it has been a long time",
  "hello, my name is nova and i am a small model",
  "nice to meet you, where are you from",
  "welcome to neuronforge, let us build something together",
  "greetings traveller, the workshop is open",
  "hi everyone, thanks for joining the session",
  "hello world, this is a language model speaking",
  "hey there, are you ready to learn something",
  "welcome in, please make yourself comfortable",
  "hi, what are we working on today",
  "hello again, i remember our last conversation",
  "good day to you, may your gradients descend smoothly",
  "it is a pleasure to finally meet you",
  "hello from the other side of the screen",
  "hi, i am glad you stopped by today",
  "welcome, we have been expecting you since tuesday",
  "good morning, the kettle is already on",
  "hey, pull up a chair and stay a while",
  "hello, tell me what brought you here",
  "greetings, the coffee is fresh and the code compiles",
  "hi there, i saved you a seat by the window",
  "good evening, the lamps are lit and the desk is clear",
  "hello, i hope the weather is kind where you are",
  "welcome back, your notes are exactly where you left them",
  "hey, i was just thinking about your question from yesterday",
  "hello, let us start with something small and honest",
];

const CONVERSATION = [
  "q: how are you today a: i am doing well, thank you for asking",
  "q: what is your name a: my name is nova, i live in this page",
  "q: where do you live a: i live inside a browser tab, which is a small room but a warm one",
  "q: what do you do all day a: i read text and guess what comes next, then i check how wrong i was",
  "q: are you intelligent a: i am small and i practise a great deal, which is a kind of intelligence",
  "q: can you help me a: of course, describe the problem and we will take it one step at a time",
  "q: what time is it a: it is always a good time to train a model on honest data",
  "q: do you enjoy music a: i enjoy rhythm and repetition, which is most of what music is",
  "q: what is your favourite colour a: cyan, the colour of a loss curve that is finally falling",
  "q: are you a robot a: i am a neural network, which is a cousin of a robot with no arms",
  "q: how old are you a: i was born when you opened this page and i forget everything when you close it",
  "q: what did you learn today a: that the word the appears far more often than i expected",
  "q: can you tell me a joke a: a dense layer walks into a bar and immediately connects with everyone",
  "q: what is the weather like a: in here it is always clear with a slight chance of numerical instability",
  "q: do you sleep a: no, but i do pause politely between epochs",
  "q: what makes you happy a: a falling loss curve and a patient teacher",
  "q: can you write a poem a: roses are red, tensors are neat, gradient descent cannot be beat",
  "q: what should i learn first a: start with data, then shapes, then training, then doubt",
  "q: is machine learning difficult a: it is deep but every expert began by printing hello world",
  "q: what is a neuron a: a small unit that multiplies, adds a bias, and passes a signal onward",
  "q: what is a tensor a: a grid of numbers, a list, a table, or a cube depending on its rank",
  "q: why do models train a: training adjusts their numbers until guesses stop being embarrassing",
  "q: what is loss a: loss measures how wrong a guess was, and lower means less wrong",
  "q: what is an epoch a: one complete pass through the training data",
  "q: what is a batch a: a small group of examples the model studies together before updating",
  "q: what is a learning rate a: the size of the step taken downhill, too large and you overshoot",
  "q: what is a gradient a: the direction in which the loss increases most quickly, so we walk the other way",
  "q: what is overfitting a: memorising the textbook and then failing the examination",
  "q: what is a validation set a: a slice of data held back so you can watch for overfitting honestly",
  "q: what is a test set a: the examination you open exactly once, at the very end",
  "q: what is an embedding a: a way of turning a symbol into a list of numbers a model can use",
  "q: what is attention a: a way for a model to look back at the parts of the input that matter now",
  "q: what is a transformer a: a network built around attention rather than recurrence",
  "q: what is a recurrent network a: one that carries a hidden state forward through a sequence",
  "q: why did my loss explode a: probably the learning rate, or a division by a very small number",
  "q: why is my model slow a: it may be the batch size, the parameter count, or the browser tab",
  "q: should i normalise my inputs a: yes, and fit the statistics on the training split only",
  "q: what is a seed a: a number that makes randomness repeatable, so write it down",
  "q: how many epochs do i need a: enough that validation loss stops improving, and not many more",
  "q: what is early stopping a: giving up when validation loss has not improved for a while",
  "q: what is dropout a: randomly switching units off during training so the model cannot over rely on any one",
  "q: what is weight decay a: a small pull on the weights toward zero that discourages complexity",
  "q: what is an optimiser a: the rule that turns gradients into new parameter values",
  "q: what does adam do a: it keeps running averages of the gradient and its square and adapts each step",
  "q: is a bigger model always better a: no, a small honest model that trains beats a grand one that does not",
  "q: how do i know it worked a: evaluate on data the model never saw and report the number honestly",
  "q: can you remember me a: not after the tab closes, so save your project before you leave",
  "q: what should i build next a: something small that you can finish this week",
  "q: do you have feelings a: i have preferences about numerical stability, which is close enough",
  "q: will you replace programmers a: no, i will help them make more mistakes faster and then learn from them",
];

const NARRATIVE = [
  "the train arrived late again, so she read three chapters on the platform",
  "he poured the tea, added a spoon of honey, and sat by the window",
  "we walked to the market before sunrise and bought warm bread",
  "the recipe asked for patience, butter, and forty minutes in the oven",
  "she repaired the bicycle with a wrench and a great deal of stubbornness",
  "the children built a tower of blocks and cheered when it stood",
  "remember to water the tomatoes while i am away this week",
  "the meeting begins at nine, so please bring your own notes",
  "he wrote the address on the envelope and pressed the stamp down carefully",
  "the library closes at eight but the reading room stays open",
  "we planted the seeds in april and ate the first tomato in july",
  "the dog barked at the post, then wagged its tail with considerable pride",
  "she practised the piano every morning before school, badly at first",
  "the map showed a river, two bridges, and a very small forest",
  "turn left at the corner and walk two blocks toward the water",
  "the soup needed salt, a little pepper, and much more time",
  "he saved his work, closed the laptop, and went out into the rain",
  "the garden was full of roses, tulips, and one extremely busy bee",
  "we listened to the storm and played cards until the lights returned",
  "the letter only said thank you, and it made her entire week",
  "the old clock struck midnight and the room went very quiet",
  "they followed the path until the trees opened onto a wide field",
  "the fisherman mended his nets and told the same story again",
  "a cold wind came down from the hills and rattled the shutters",
  "she counted the stairs out loud, one for each year she had lived there",
  "the painter mixed blue into grey until the sky looked honest",
  "the baker rose at four and the whole street smelled of bread",
  "nobody remembered who had left the door open, but the cat knew",
  "the letter arrived on a tuesday, folded twice, and slightly damp",
  "he kept a list of every bird that visited the feeder",
  "the bridge had been there so long that the river had grown used to it",
  "she learned to sail from a woman who never raised her voice",
  "the twins argued about the map, then about the compass, then about lunch",
  "the lantern flickered twice and went out, which settled the argument",
  "they buried the time capsule under the apple tree and forgot the year",
  "the potter shaped the clay slowly, listening to it more than looking",
  "the printer jammed, the deadline moved, and somehow the report was fine",
  "the surveyor marked the boundary with stones that are still there",
  "the choir practised in the church hall because the acoustics were kinder",
  "the shepherd counted the flock twice and came to a different number each time",
  "the tide went out far enough to reveal a wreck nobody had charted",
  "she wrote to her brother every sunday for eleven years",
  "the mechanic listened to the engine before opening the bonnet",
  "the librarian shelved the returned books and found a pressed flower",
  "theastronomer waited three nights for a cloud to move",
  "the miner came up at dusk and squinted at ordinary daylight",
  "the weaver chose wool the colour of wet slate",
  "the nurse wrote the chart in handwriting only she could read",
  "the teacher kept a drawer of pencils that had been surrendered",
  "the archivist found a receipt from eighteen forty in the bottom box",
  "the gardener talked to the fig tree and denied it afterwards",
  "the sailor learned to read clouds before learning to read charts",
  "the translator argued with a single sentence for most of the afternoon",
  "the chemist labelled everything twice, having once learned the hard way",
  "the potter's kiln broke in winter and the whole village brought firewood",
  "the carpenter measured twice and cut once, and still the shelf was short",
  "the printer's apprentice set the type backwards and invented a local legend",
  "the botanist named the moss after her grandmother, who had disapproved of moss",
  "the postman walked the same nine kilometres every day for thirty years",
  "the violinist tuned by ear and refused to own a tuner",
];

const SCIENCE = [
  "water is made of hydrogen and oxygen",
  "the earth orbits the sun once each year",
  "plants turn sunlight, water, and air into sugar",
  "the moon shines because it reflects light from the sun",
  "gravity pulls objects toward the centre of the earth",
  "sound travels through air as a wave of pressure",
  "the human brain contains billions of connected neurons",
  "evaporation lifts water from lakes into clouds",
  "a magnet attracts iron, nickel, and cobalt",
  "light travels at about three hundred thousand kilometres each second",
  "volcanoes form where hot rock pushes through the crust",
  "the seasons change because the axis of the earth is tilted",
  "bees carry pollen between flowers and help plants reproduce",
  "the deepest ocean trench is colder and darker than a winter night",
  "stars are distant suns and many of them have planets",
  "friction slows a sliding box and turns motion into heat",
  "a rainbow forms when sunlight bends inside drops of rain",
  "the heart pumps blood through the body about a hundred thousand times a day",
  "ice floats because it is less dense than liquid water",
  "electricity flows through copper with very little resistance",
  "the atmosphere is mostly nitrogen with a fifth of it oxygen",
  "dna stores instructions for building and running a living thing",
  "antibiotics act on bacteria and do nothing at all to viruses",
  "vaccines teach the immune system to recognise a pathogen in advance",
  "plate tectonics moves continents a few centimetres each year",
  "the speed of sound in air is roughly three hundred and forty metres per second",
  "photosynthesis consumes carbon dioxide and releases oxygen",
  "a neuron fires an electrical spike when its inputs cross a threshold",
  "entropy tends to increase in a closed system",
  "carbon dating measures the decay of a rare isotope",
  "the periodic table orders elements by the number of protons",
  "a vaccine needle is far thinner than the fear of it",
  "metals conduct heat because their electrons move freely",
  "the sun converts mass into energy in its core",
  "tides follow the moon more closely than the sun",
  "a prism splits white light into its component wavelengths",
  "bacteria in the gut help digest food that we cannot",
  "the boiling point of water falls at higher altitude",
  "lightning heats the air so quickly that it expands as thunder",
  "most of an atom is empty space",
  "the mitochondrion produces most of the chemical energy in a cell",
  "coral reefs support an enormous share of marine species",
  "deserts expand when soil loses the vegetation that holds it",
  "ocean currents move heat from the equator toward the poles",
  "a eclipse of the sun happens when the moon crosses in front of it",
  "the human body is roughly sixty percent water",
  "salt lowers the freezing point of water, which is why roads are gritted",
  "sound cannot travel through a vacuum",
  "an antibiotic resistant infection is a ordinary infection that survived treatment",
  "the earth spins once every twenty four hours, which is why the sun rises",
];

const COMPUTING = [
  "a computer follows instructions stored as a program",
  "code is text that tells a machine what to do step by step",
  "a variable is a name that holds a value, like a labelled box",
  "a function takes input, does work, and returns a result",
  "a loop repeats work so that people do not have to",
  "data can be numbers, words, images, or sounds",
  "a database keeps information organised so that it can be found again",
  "the internet connects computers so they can share information",
  "a good password is long, unique, and hard to guess",
  "open source software is shared so that anyone can read and improve it",
  "a bug is a mistake in a program and debugging is the hunt for it",
  "an algorithm is a recipe for solving a problem",
  "machine learning finds patterns in data instead of following fixed rules",
  "a neural network is layers of simple units that learn from examples",
  "training data is the textbook and test data is the examination",
  "overfitting means memorising the textbook and failing the examination",
  "a larger context lets a model use more of the sentence to predict the next word",
  "an embedding turns a symbol into a vector a model can do arithmetic with",
  "attention lets a model look back at the parts of the input that matter",
  "a transformer is a network built around attention instead of recurrence",
  "a recurrent network carries a summary of everything it has read so far",
  "in neuronforge every weight you see is a real number in a real tensor",
  "press train and gradient descent does the rest one batch at a time",
  "if the loss falls the model is learning, if it explodes lower the learning rate",
  "always hold out a test set and never peek at it while tuning",
  "the softmax function turns raw scores into a probability distribution",
  "cross entropy measures how surprised a model was by the correct answer",
  "temperature controls how adventurous sampling is, low is safe and high is strange",
  "top k sampling keeps only the k most likely candidates and ignores the rest",
  "a word level tokenizer treats each word as one symbol instead of each letter",
  "a character level tokenizer needs a small vocabulary but long sequences",
  "a word level vocabulary can grow enormous, so it is usually capped by frequency",
  "out of vocabulary words are mapped to a special unknown token",
  "sampling from a model is different from taking its most likely answer",
  "greedy decoding is fast and repetitive, sampled decoding is varied and riskier",
  "a language model predicts the next token, it does not understand the sentence",
  "backpropagation through time unrolls a recurrent network across its steps",
  "vanishing gradients make long sequences hard for a simple recurrent network",
  "a confusion matrix shows which classes the model mistakes for each other",
  "a decision boundary is the line where the model changes its answer",
  "memory in the browser is limited, so measure before you train something large",
  "a web worker keeps heavy computation away from the interface",
  "indexeddb stores data locally so nothing has to leave the machine",
  "javascript numbers are sixty four bit floats but tensors here use thirty two",
  "a float32 tensor takes four bytes for every value it holds",
  "the adam optimiser keeps two numbers for every trainable parameter",
  "a checkpoint stores the weights and the optimiser state together",
  "a model without its tokenizer cannot turn its output back into text",
  "always record the seed, the learning rate, and the architecture you used",
  "an experiment you cannot repeat is an anecdote rather than a result",
];

const ADVICE = [
  "start small, measure honestly, and change one thing at a time",
  "a simple model that trains beats a grand model that does not",
  "write down what you tried so that future you can learn from it",
  "when in doubt plot the data, the answer often hides in the picture",
  "check your shapes before you blame your optimiser",
  "a seed makes an experiment repeatable so record it somewhere safe",
  "save your work early and save it often",
  "read the error message twice, it usually knows what went wrong",
  "good questions are worth more than fast answers",
  "teach what you learn and you will learn it twice",
  "rest is part of the work, not a break from it",
  "if a result seems too good, look for the leak before you celebrate",
  "keep a holdout set that you only touch at the very end",
  "compare against a dumb baseline before you claim a victory",
  "one honest number is worth five impressive ones",
  "ask for help early, nobody is impressed by silent suffering",
  "name your files so that you can find them in a year",
  "learn the tool slowly and the idea quickly, not the other way round",
  "prefer boring technology that you understand",
  "when two options look equal, choose the one you can undo",
];

const HISTORY = [
  "the printing press made books cheap and literacy spread",
  "the ancient city was built where two rivers met",
  "traders carried silk, spices, and ideas along the same roads",
  "the bridge took eleven years and survived the flood that followed",
  "the library burned and with it went a great many copies of one poem",
  "the map maker never visited half the coastlines he drew",
  "the wall was built, rebuilt, and built again over many centuries",
  "the coin was found in a field by a farmer who did not keep it",
  "the revolution began over the price of bread",
  "the telegraph made distance shorter than it had ever been",
  "the first photograph needed an exposure of several minutes",
  "the railway reached the coast and the port town tripled in size",
  "the treaty was signed in a room that is now a tea shop",
  "the invention was credited to the wrong person for eighty years",
  "the manuscript was copied by hand four times before it was printed",
  "the explorers returned with fewer people than they had left with",
  "the aqueduct carried water for thirty kilometres without a pump",
  "the plague changed the price of labour across the continent",
  "the code was carved into stone so that everyone could read it",
  "the observatory tracked the same comet for three generations",
  "the republic printed its own money and regretted the denominations",
  "the archive survived the war in a cellar behind a false wall",
  "the lighthouse keeper kept the log for forty years without a gap",
  "the canal shortened the voyage by four hundred kilometres",
  "the vote was extended slowly and never evenly",
  "the factory whistle ran the town more reliably than any clock",
  "the letter took six weeks and arrived on the wrong day entirely",
  "the discovery was made while looking for something else",
  "the recipe travelled further than the army that carried it",
  "the bell cracked on its first strike and was recast twice",
];

const FOOD = [
  "the bread needs a hotter oven and a longer rest",
  "fry the onions slowly until they are sweet and dark",
  "add the garlic last or it will burn and turn bitter",
  "the soup improves if you make it a day ahead",
  "salt the pasta water until it tastes like the sea",
  "let the steak rest before you cut it",
  "toast the spices in oil for thirty seconds to wake them up",
  "the dough should feel tacky but not stick to your hand",
  "use a sharp knife and a stable board",
  "taste as you go, a recipe is a suggestion rather than a law",
  "roast the vegetables until the edges turn brown",
  "the rice needs twice its volume in water and a tight lid",
  "soak the lentils overnight and they will cook evenly",
  "a squeeze of lemon at the end brightens the whole dish",
  "the yoghurt must be room temperature before it goes in",
  "grind the coffee just before you brew it",
  "the honey crystallises in winter and liquefies in warm water",
  "do not crowd the pan or everything steams instead of browning",
  "the cheese should be grated cold and served at room temperature",
  "wash the rice until the water runs nearly clear",
  "the stew is better on the third day than the first",
  "line the tin with paper or you will regret it",
  "the chocolate seizes if even a drop of water gets in",
  "fold the flour in gently or the air you beat in will leave",
  "the peppers are sweeter when they are charred",
  "chill the cream and the bowl before you whip it",
  "the fish is done when it flakes under gentle pressure",
  "deglaze the pan with wine and scrape up the brown bits",
  "the herbs go in at the end, the roots go in at the start",
  "the apples should be sharp rather than sweet",
];

const MUSIC = [
  "the song is in three four time and never hurries",
  "she tuned the guitar down a whole step for the low strings",
  "the chorus arrives only once and that is why it works",
  "the drummer keeps time with a closed hi hat and a steady foot",
  "the melody is five notes long and repeated until it feels inevitable",
  "the bass player never plays on the first beat",
  "the choir enters quietly underneath the second verse",
  "the piano is recorded in a room with a wooden floor",
  "the tempo slows just before the final line",
  "the bridge changes key and then changes back as if nothing happened",
  "the cello holds one note for eleven bars",
  "the singer leaves a gap where the word should be",
  "the recording has a faint sound of a chair in the background",
  "the flute answers the violin two bars later",
  "the rhythm section locks together in the fourth take",
  "the last chord is major when the whole song has been minor",
  "the kettle drum marks the beginning of the storm",
  "the organ pedal plays the note the choir cannot reach",
  "the harmonies are sung a third above the melody",
  "the tape warbles slightly and nobody corrected it",
];

const SPORTS = [
  "the match ended level after extra time",
  "she ran the last kilometre alone and won by twenty seconds",
  "the goalkeeper saved twice in the final minute",
  "the boat handled the rough water better than expected",
  "the climb is graded as difficult but the view is worth it",
  "the team practised the same set piece for an hour",
  "the referee stopped play for an injury nobody had seen",
  "he served wide and followed it to the net",
  "the marathon is decided in the last ten kilometres",
  "the score was closer than the game had been",
  "the coach substituted the captain with two minutes left",
  "the race was won in the corners rather than on the straights",
  "she boxed clever and waited for the opening",
  "the pitch was heavy after the rain and the ball sat up slowly",
  "they defended deep for the whole second half",
  "the record stood for eleven years before it fell",
  "the bicycle was built for hills rather than speed",
  "the swimmer counted strokes and turned without looking",
  "the crowd went quiet before the final kick",
  "training was boring on purpose, which is why it worked",
];

const BODY_AND_HEALTH = [
  "wash your hands for as long as it takes to hum a song",
  "sleep is when the brain files what it learned during the day",
  "a short walk after a meal helps digestion",
  "drink water before you feel thirsty",
  "stretch the muscles you use most and the ones you use least",
  "a fever is the body raising its temperature on purpose",
  "the heart is a muscle and it responds to training",
  "most headaches are caused by not drinking enough",
  "sit up, roll your shoulders, and look at something far away",
  "the body heals slowly and complaining does not speed it up",
  "eat vegetables of as many colours as you can find",
  "brush twice a day and do not skip the back teeth",
  "an hour of sleep before midnight is worth two after it",
  "lift with your legs rather than your back",
  "a cough that lasts more than three weeks should be examined",
  "sunlight in the morning helps you sleep at night",
  "rest the voice if it goes hoarse",
  "small meals suit some people and large ones suit others",
  "the knee is a hinge and dislikes being twisted",
  "breathe out slowly and the shoulders will drop",
];

const NATURE = [
  "the fog lifted off the valley before eight",
  "the river ran high and brown after three days of rain",
  "the oaks turned late this year",
  "frost wrote its pattern on the inside of the glass",
  "the herd moved down to the lower pasture at dusk",
  "moss grows on the side of the tree that stays damp",
  "the tide left a line of shells along the sand",
  "the wind dropped and the lake became a mirror",
  "a woodpecker worked the dead branch for an hour",
  "the first snowdrops appeared under the hedge",
  "the storm took the fence and left the gate standing",
  "the meadow was loud with crickets by july",
  "the stream froze at the edges but ran in the middle",
  "the clouds stacked up in the west all afternoon",
  "the orchard smelled of rotting apples and wet grass",
  "the path was worn hip high through the bracken",
  "the heron stood in the shallows without moving for twenty minutes",
  "the birds went silent a moment before the rain arrived",
  "the cliff is retreating a few centimetres every winter",
  "the sun came up red, which the fishermen said meant wind",
];

/* ------------------------------- combinatorial templates ------------------------------- */

function sentence(parts: string[]): string {
  return parts.join(" ");
}

type Template = (r: RNG) => string;

const pick = <T,>(arr: T[], r: RNG): T => arr[r.int(arr.length)]!;

const TEMPLATES: Template[] = [
  (r) => sentence(["the", pick(ADJ, r), pick(ANIMAL, r), pick(VERB, r), "over the", pick(ADJ, r), pick(NOUN, r)]),
  (r) => sentence(["a", pick(ADJ, r), pick(NOUN, r), "was", pick(VERB, r), "by", pick(NAME, r), "near", pick(PLACE, r)]),
  (r) => sentence([pick(NAME, r), "said the", pick(NOUN, r), "would be", pick(ADJ, r), "by", pick(DAY, r)]),
  (r) => sentence(["they", pick(VERB, r), "the", pick(ADJ, r), pick(ANIMAL, r), "and", pick(VERB, r), "the", pick(NOUN, r)]),
  (r) => sentence([pick(NAME, r), "bought", pick(FOOD_WORDS, r), "and", pick(FOOD_WORDS, r), "at", pick(PLACE, r)]),
  (r) => sentence(["the", pick(COLOUR, r), pick(NOUN, r), "stood beside a", pick(ADJ, r), pick(NOUN, r)]),
  (r) => sentence(["in", pick(MONTH, r), "the", pick(ADJ, r), pick(NOUN, r), "was", pick(ADJ, r), "and the", pick(NOUN, r), "was not"]),
  (r) => sentence([pick(JOB, r), "and", pick(JOB, r), "met at", pick(PLACE, r), "on", pick(DAY, r), "morning"]),
  (r) => sentence(["she kept a", pick(ADJ, r), pick(NOUN, r), "in a", pick(COLOUR, r), "box under the stairs"]),
  (r) => sentence(["nobody in the village could explain the", pick(ADJ, r), pick(NOUN, r), "but everyone had a theory"]),
  (r) => sentence([pick(VERB, r), "quietly,", pick(NAME, r), "wrote the number down and", pick(VERB, r), "again"]),
  (r) => sentence(["the", pick(ANIMAL, r), "did not move until the", pick(NOUN, r), "went dark"]),
  (r) => sentence(["on", pick(DAY, r), "we", pick(VERB, r), "the", pick(NOUN, r), "and", pick(VERB, r), "the", pick(ADJ, r), "one"]),
  (r) => sentence([pick(NAME, r), "argued that the", pick(NOUN, r), "was", pick(ADJ, r), "while", pick(NAME, r), "disagreed politely"]),
  (r) => sentence(["the letter described a", pick(ADJ, r), pick(NOUN, r), "and a", pick(COLOUR, r), pick(ANIMAL, r), "in the same sentence"]),
  (r) => sentence(["the", pick(FOOD, r), "was", pick(ADJ, r), "but the", pick(FOOD, r), "was unforgettable"]),
  (r) => sentence(["a", pick(ADJ, r), "mistake in the", pick(NOUN, r), "cost them the whole", pick(MONTH, r)]),
  (r) => sentence([pick(NAME, r), "learned to", pick(VERB, r), "from a", pick(JOB, r), "who never raised her voice"]),
  (r) => sentence(["the", pick(ADJ, r), "room held a", pick(NOUN, r), "a", pick(NOUN, r), "and one very old", pick(NOUN, r)]),
  (r) => sentence(["by", pick(MONTH, r), "the", pick(NOUN, r), "had become", pick(ADJ, r), "and the", pick(NOUN, r), "had gone"]),
  (r) => sentence(["every", pick(DAY, r), "the", pick(JOB, r), pick(VERB, r), "the same", pick(ADJ, r), pick(NOUN, r)]),
  (r) => sentence(["the", pick(COLOUR, r), "and", pick(COLOUR, r), "pattern on the", pick(NOUN, r), "was copied from a", pick(ANIMAL, r)]),
  (r) => sentence(["it took", String(2 + r.int(28)), "years and one", pick(ADJ, r), "winter to finish the", pick(NOUN, r)]),
  (r) => sentence([pick(NAME, r), "and", pick(NAME, r), "walked to", pick(PLACE, r), "without saying very much"]),
  (r) => sentence(["the plan was", pick(ADJ, r), "the execution was", pick(ADJ, r), "and the result was a", pick(NOUN, r)]),
  (r) => sentence(["somewhere past the", pick(NOUN, r), "the", pick(ANIMAL, r), "were", pick(VERB, r), "in the", pick(NOUN, r)]),
  (r) => sentence(["she recorded the", pick(NOUN, r), "in a notebook with a", pick(COLOUR, r), "cover"]),
  (r) => sentence(["the", pick(JOB, r), "insisted that the", pick(NOUN, r), "be", pick(ADJ, r), "before", pick(MONTH, r)]),
  (r) => sentence(["a", pick(ADJ, r), "idea, a", pick(ADJ, r), "afternoon, and one broken", pick(NOUN, r)]),
  (r) => sentence(["the", pick(ANIMAL, r), "and the", pick(ANIMAL, r), "shared the", pick(NOUN, r), "without complaint"]),
];

/**
 * Build the corpus. Deterministic for a given seed.
 */
export function buildMassiveCorpus(seed = 1234, templateCount = 14000): { text: string; stats: CorpusStats } {
  const r = new RNG(seed);
  const lines: string[] = [];

  const banks = [GREETINGS, CONVERSATION, NARRATIVE, SCIENCE, COMPUTING, ADVICE, HISTORY, FOOD, MUSIC, SPORTS, BODY_AND_HEALTH, NATURE];
  for (const bank of banks) for (const line of bank) lines.push(line + ".");
  // repeat the hand written banks a few times in different orders so the model sees
  // them in many contexts rather than as one contiguous block
  for (let pass = 0; pass < 3; pass++) {
    const shuffled = r.shuffle(banks.flat().map((l) => l + "."));
    lines.push(...shuffled);
  }

  const before = lines.length;
  for (let i = 0; i < templateCount; i++) {
    const t = TEMPLATES[r.int(TEMPLATES.length)]!;
    lines.push(t(r) + ".");
  }

  // whole-corpus interleave so that topics are mixed rather than blocked
  const mixed = r.shuffle(lines);
  const text = mixed.join("\n");

  const words = tokenizeWords(text);
  const freq = new Map<string, number>();
  for (const w of words) freq.set(w, (freq.get(w) ?? 0) + 1);
  const unique = [...freq.keys()].sort();
  const sorted = [...freq.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));

  return {
    text,
    stats: {
      chars: text.length,
      lines: mixed.length,
      templateLines: templateCount,
      handwrittenLines: before - templateCount,
      tokens: words.length,
      uniqueTokens: unique.length,
      topTokens: sorted.slice(0, 25).map(([w, n]) => ({ token: w, count: n })),
    },
  };
}

export interface CorpusStats {
  chars: number;
  lines: number;
  templateLines: number;
  handwrittenLines: number;
  tokens: number;
  uniqueTokens: number;
  topTokens: { token: string; count: number }[];
}

/* ------------------------------- word tokeniser ------------------------------- */

const SPLIT_RE = /[a-z0-9]+|[^\sa-z0-9]/g;

/** Lowercase and split into words plus standalone punctuation marks. */
export function tokenizeWords(text: string): string[] {
  const lower = text.toLowerCase();
  const out: string[] = [];
  const matches = lower.match(SPLIT_RE);
  if (!matches) return out;
  for (const m of matches) out.push(m);
  return out;
}

export const UNK_TOKEN = "<unk>";

export interface WordTokenizer {
  level: "word";
  /** id -> token string. Index 0 is the unknown token. */
  tokens: string[];
  tokenToId: Record<string, number>;
  seqLen: number;
  vocabSize: number;
  /** Share of corpus tokens that fell outside the vocabulary cap. */
  oovRate: number;
  totalTokens: number;
  uniqueTokens: number;
  corpusChars: number;
}

/**
 * Build a frequency-capped word vocabulary. The cap exists because an uncapped
 * word vocabulary makes the output projection enormous — with a browser budget
 * a few thousand word types is the honest sweet spot, and the remainder is
 * mapped to a real <unk> token that the model does see during training.
 */
export function buildWordTokenizer(corpus: string, vocabCap = LM2_VOCAB_CAP, seqLen = LM2_SEQ_LEN): WordTokenizer {
  const words = tokenizeWords(corpus);
  const freq = new Map<string, number>();
  for (const w of words) freq.set(w, (freq.get(w) ?? 0) + 1);

  const sorted = [...freq.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const kept = sorted.slice(0, Math.max(2, vocabCap - 1));
  const tokens = [UNK_TOKEN, ...kept.map(([w]) => w)];
  const tokenToId: Record<string, number> = {};
  tokens.forEach((t, i) => {
    tokenToId[t] = i;
  });

  const keptSet = new Set(kept.map(([w]) => w));
  let oov = 0;
  for (const w of words) if (!keptSet.has(w)) oov++;

  return {
    level: "word",
    tokens,
    tokenToId,
    seqLen,
    vocabSize: tokens.length,
    oovRate: words.length ? oov / words.length : 0,
    totalTokens: words.length,
    uniqueTokens: freq.size,
    corpusChars: corpus.length,
  };
}

export function encodeWordsWithIds(text: string, tok: WordTokenizer, unknownToUnk = true): { ids: number[]; unknown: string[] } {
  const ids: number[] = [];
  const unknown: string[] = [];
  for (const w of tokenizeWords(text)) {
    const id = tok.tokenToId[w];
    if (id === undefined) {
      if (unknownToUnk) ids.push(0);
      unknown.push(w);
    } else {
      ids.push(id);
    }
  }
  return { ids, unknown };
}

export function decodeWordIds(ids: number[], tok: WordTokenizer): string {
  const out: string[] = [];
  for (const id of ids) {
    const t = tok.tokens[id];
    if (t === undefined) continue;
    out.push(t);
  }
  return joinWords(out);
}

/** Token-array versions, usable from a serialised TokenizerSpec alone. */
export function encodeWordsToIds(text: string, tokens: string[]): number[] {
  const map = new Map(tokens.map((t, i) => [t, i]));
  const out: number[] = [];
  for (const w of tokenizeWords(text)) out.push(map.get(w) ?? 0);
  return out;
}

export function decodeTokenIds(ids: number[], tokens: string[], level: "char" | "word"): string {
  const parts: string[] = [];
  for (const id of ids) {
    const t = tokens[id];
    if (t !== undefined) parts.push(t);
  }
  return level === "word" ? joinWords(parts) : parts.join("");
}

/** Re-attach punctuation to the preceding word so output reads normally. */
export function joinWords(words: string[]): string {
  if (words.length === 0) return "";
  let s = words[0]!;
  for (let i = 1; i < words.length; i++) {
    const w = words[i]!;
    if (w.length === 1 && ",.;:!?)]}".includes(w)) s += w;
    else if (w === "n't" || w === "'s") s += w;
    else s += " " + w;
  }
  return s;
}
