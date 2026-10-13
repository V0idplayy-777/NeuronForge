/**
 * Built-in language corpus for the tiny character models.
 *
 * The tokenizer is *derived from this text*: TEXT_CHARS is the sorted set of
 * unique characters, and every model/dataset built for language work reads its
 * vocabulary size and char<->id mapping from here. If you edit the corpus, the
 * vocabulary changes everywhere consistently — and the mapping is written into
 * exported NeuronForge model files so a saved model is never reinterpreted
 * against a different alphabet.
 *
 * Style rules for this corpus: lowercase only, plain ascii punctuation, one
 * utterance per line. Newline is a real token — it teaches the model where
 * utterances end.
 */

export const LM_SEQ_LEN = 64;

const GREETINGS = [
  "hello there, how are you today?",
  "hi friend, it is good to see you.",
  "good morning! did you sleep well?",
  "good afternoon, what can i do for you?",
  "good evening, welcome back home.",
  "hey, long time no see. how have you been?",
  "hello, my name is nova. what is your name?",
  "nice to meet you. where are you from?",
  "welcome to neuronforge. let us build something.",
  "greetings, traveler. the lab is open.",
  "hi everyone, thanks for joining us today.",
  "hello world, this is a tiny language model speaking.",
  "good day to you. may your gradients descend smoothly.",
  "hey there. ready to learn something new?",
  "hello again! i remember our last conversation.",
  "welcome in. please make yourself comfortable.",
  "hi! what are we working on today?",
  "hello, hello. the coffee is fresh and the code compiles.",
  "good morning, good morning. the sun is up.",
  "hey friend, pull up a chair and stay a while.",
  "it is a pleasure to meet you at last.",
  "hello from the other side of the screen.",
  "hi, i am glad you stopped by.",
  "welcome, welcome. we have been expecting you.",
];

const SMALL_TALK = [
  "q: how are you? a: i am doing well, thank you for asking.",
  "q: what is your name? a: my name is nova, a small model in a browser.",
  "q: where do you live? a: i live in this page, inside your computer.",
  "q: what do you do? a: i learn patterns in text, one character at a time.",
  "q: are you smart? a: i am tiny, but i practice every day.",
  "q: can you help me? a: of course, tell me what you need.",
  "q: what time is it? a: it is always time to train a model.",
  "q: do you like music? a: i like rhythm, repetition, and a good beat.",
  "q: what is your favorite color? a: cyan, the color of a fresh loss curve.",
  "q: are you a robot? a: i am a neural network, which is close enough.",
  "q: how old are you? a: i was born the moment you opened this page.",
  "q: what did you learn today? a: i learned that hello often comes before world.",
  "q: can you tell a joke? a: why did the neuron cross the layer? to get activated.",
  "q: what is the weather like? a: in here it is always clear with a chance of gradients.",
  "q: do you sleep? a: no, but i do pause between epochs.",
  "q: what makes you happy? a: a falling loss and a kind user.",
  "q: can you write a poem? a: roses are red, tensors are neat, gradient descent cannot be beat.",
  "q: what should i learn first? a: start with data, then shapes, then training.",
  "q: is machine learning hard? a: it is deep, but every expert began with hello world.",
  "q: what is a neuron? a: a small unit that multiplies, adds, and passes a signal on.",
  "q: what is a tensor? a: a tensor is a grid of numbers, like a list or a table or a cube.",
  "q: why do models train? a: training adjusts their numbers until guesses become good.",
  "q: what is loss? a: loss measures how wrong a guess is. lower is better.",
  "q: what is an epoch? a: one full pass through the training data.",
  "q: what is a batch? a: a small group of examples the model studies at once.",
];

const EVERYDAY = [
  "the train is late again, so i read a book on the platform.",
  "she poured the tea, added a spoon of honey, and smiled.",
  "we walked to the market and bought bread, cheese, and apples.",
  "the recipe says to bake the bread for forty minutes.",
  "he fixed the bicycle with a wrench, patience, and youtube-free intuition.",
  "the children built a tower of blocks and cheered when it stood tall.",
  "remember to water the plants while i am away this weekend.",
  "the meeting starts at nine, so please bring your notes.",
  "i wrote the address on the envelope and added a stamp.",
  "the library closes at eight, but the reading room stays open late.",
  "we planted tomatoes in april and ate the first one in july.",
  "the dog barked at the mail, then wagged its tail with pride.",
  "she practiced the piano every morning before school.",
  "the map shows a river, two bridges, and a small forest.",
  "please turn left at the corner and walk two blocks north.",
  "the soup needs salt, a little pepper, and more time.",
  "he saved his work, closed the laptop, and went for a walk.",
  "the garden is full of roses, tulips, and one very busy bee.",
  "we listened to the rain and played cards by the window.",
  "the letter said thank you, and it made her whole day.",
];

const SCIENCE_AND_WORLD = [
  "water is made of hydrogen and oxygen.",
  "the earth orbits the sun once every year.",
  "plants turn sunlight, water, and air into food.",
  "the moon shines because it reflects light from the sun.",
  "gravity pulls objects toward the center of the earth.",
  "sound travels through air as a wave of pressure.",
  "the human brain has billions of neurons connected in networks.",
  "evaporation lifts water from lakes into the sky, where clouds form.",
  "a magnet attracts iron, nickel, and cobalt.",
  "the speed of light is about three hundred thousand kilometers per second.",
  "volcanoes form where hot rock pushes through the crust of the earth.",
  "the seasons change because the axis of the earth is tilted.",
  "bees carry pollen from flower to flower and help plants grow.",
  "the deepest ocean is darker and colder than a winter night.",
  "stars are distant suns, and some have planets of their own.",
  "friction slows a sliding box and turns motion into heat.",
  "a rainbow appears when sunlight passes through drops of rain.",
  "the heart pumps blood through the body with every beat.",
  "ice floats on water because it is less dense when frozen.",
  "electricity flows through copper wire with very little resistance.",
  "the amazon rainforest produces a large share of the oxygen on earth.",
  "mount everest is the tallest mountain above sea level.",
  "the sahara is the largest hot desert in the world.",
  "a year on mars lasts almost twice as long as a year on earth.",
  "the great wall of china stretches for thousands of kilometers.",
];

const COMPUTING = [
  "a computer follows instructions stored as programs.",
  "code is text that tells a machine what to do, step by step.",
  "a variable is a name that holds a value, like a labeled box.",
  "a function takes input, does work, and returns a result.",
  "a loop repeats work so people do not have to.",
  "data can be numbers, words, images, or sounds.",
  "a database keeps information organized so it can be found again.",
  "the internet connects computers so they can share information.",
  "a password should be long, unique, and hard to guess.",
  "open source software is shared so everyone can learn from it.",
  "a bug is a mistake in a program, and debugging is the hunt for it.",
  "an algorithm is a recipe for solving a problem.",
  "machine learning finds patterns in data instead of following fixed rules.",
  "a neural network is layers of simple units that learn from examples.",
  "training data is the textbook, and testing data is the exam.",
  "overfitting means memorizing the textbook and failing the exam.",
  "a larger context lets a model use more of the sentence to predict the next word.",
  "an embedding turns a symbol into a list of numbers a model can use.",
  "attention lets a model look back at the parts of the input that matter.",
  "a transformer is a network built around attention instead of recurrence.",
  "in neuronforge, every weight you see is a real number in a real tensor.",
  "press train, and gradient descent will do the rest, one batch at a time.",
  "if the loss goes down, the model is learning. if it explodes, lower the learning rate.",
  "always hold out a test set, and never peek at it while you tune.",
];

const NUMBERS_AND_TIME = [
  "one, two, three, four, five, six, seven, eight, nine, ten.",
  "the days of the week are monday, tuesday, wednesday, thursday, friday, saturday, and sunday.",
  "the months of the year are january, february, march, april, may, june, july, august, september, october, november, and december.",
  "there are sixty seconds in a minute and sixty minutes in an hour.",
  "a week has seven days, and a year has fifty two weeks.",
  "colors of the rainbow are red, orange, yellow, green, blue, indigo, and violet.",
  "first we gather data, second we build the model, third we train, and fourth we test.",
  "the answer is forty two, but the question matters more.",
  "she counted the steps: ten to the door, twenty to the gate, thirty to the road.",
  "practice a little every day and the progress will surprise you.",
];

const ADVICE = [
  "start small, measure honestly, and change one thing at a time.",
  "a simple model that trains beats a grand model that does not.",
  "write down what you tried, so future you can learn from it.",
  "when in doubt, plot the data. the answer often hides in the picture.",
  "check your shapes before you blame your optimizer.",
  "a seed makes an experiment repeatable, so record it.",
  "save your work early and save it often.",
  "read the error message twice. it usually knows what went wrong.",
  "good questions are worth more than fast answers.",
  "teach what you learn, and you will learn it twice.",
];

export const TEXT_CORPUS: string = [
  ...GREETINGS,
  ...SMALL_TALK,
  ...EVERYDAY,
  ...SCIENCE_AND_WORLD,
  ...COMPUTING,
  ...NUMBERS_AND_TIME,
  ...ADVICE,
].join("\n");

/** Sorted unique characters of the corpus — the one true tokenizer. */
export const TEXT_CHARS: string[] = [...new Set(TEXT_CORPUS.split(""))].sort();

export const TEXT_VOCAB_SIZE: number = TEXT_CHARS.length;

export function encodeText(text: string, chars: string[] = TEXT_CHARS): { ids: number[]; skipped: string[] } {
  const stoi = new Map(chars.map((c, i) => [c, i]));
  const ids: number[] = [];
  const skipped: string[] = [];
  for (const ch of text.toLowerCase()) {
    const id = stoi.get(ch);
    if (id === undefined) skipped.push(ch);
    else ids.push(id);
  }
  return { ids, skipped };
}

export function decodeIds(ids: number[], chars: string[] = TEXT_CHARS): string {
  return ids.map((i) => chars[i] ?? "").join("");
}
