# We Built a Security System That Listens, and the Second Model Nearly Sank It

**By Zarwasham Lodhi, with Hamza Khan, Ibad Ur Rehman and Sarim Owais**

Cameras are blind to most of what matters.

Point one down a corridor and it has no idea a window just broke in the room
behind it. It cannot hear a scream from the stairwell, or a compressor starting
to fail in a plant room at three in the morning. Sound goes around corners.
Video does not.

That is the gap SonicSentinel AI was built for. It takes audio from a microphone
or an uploaded recording, decides what it is hearing across ten safety relevant
classes, and raises an alert when a human needs to look. Gunshot, glass
breaking, panic scream, aggression, someone calling for help, alarms, vehicle
horns, animals, machinery faults, and plain background noise.

This is the story of building it, including the part where our second model
turned out to be much worse than we thought, and what we did about it.

---

## Two models that are not allowed to talk

The core design decision was to classify everything twice.

One model runs on the server in Python. A second, entirely separate model runs
inside the browser tab. They see the same audio and neither is ever shown the
other's answer.

That sounds like redundancy for its own sake. It is not. A single model that is
confidently wrong tells you nothing about its own reliability. Two independent
models give you a signal you can actually act on. Where they agree, confidence is
high. Where they disagree, the event goes to a person instead of being quietly
guessed at.

Keeping them honest took some care. The browser posts its scores to the server
only after the server has already stored its own decision. That ordering is the
thing that makes the independence real rather than a claim in a document.

---

## The server side went well

We trained five models on the server and compared them on a held out test split
that was never touched during training, tuning or selection.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 68.09% | 0.6812 |
| Random Forest | 69.44% | 0.6904 |
| SVM | 74.38% | 0.7424 |
| XGBoost | 74.83% | 0.7455 |
| **YAMNet transfer** | **90.02%** | **0.9000** |

The winner was a small scikit-learn head trained on frozen YAMNet embeddings.
YAMNet is a pretrained audio network, and instead of training a classifier from
scratch on three thousand clips we let it do the heavy lifting and only learned
the last step.

At 90.02 percent accuracy and 0.9000 macro F1, with every one of the five
critical classes above 85 percent recall, it clears the full requirement.

That was the easy half.

---

## The browser side did not

The second model has a harder job. It has to run inside a browser tab, in
JavaScript, fast enough to keep up with a live microphone.

We started with Google Teachable Machine. It exports straight to TensorFlow.js,
which made it the obvious choice.

Then we hit the wall.

**Teachable Machine would not accept our dataset.** The platform caps how much
audio you can feed a class. In practice we could supply about 100 one second
samples per class before it would take no more. We had three thousand collected,
cleaned and labelled recordings. A class with 255 usable training recordings had
to be cut down to 100 single second excerpts, one per recording, picked by
loudness. Everything else was thrown away before training started.

You also get no control over anything that matters. No training schedule, no
class weighting, no early stopping, no way to look at what the network learned.

The results were what you would expect from a tenth of your data:

| Teachable Machine export | Accuracy | Macro F1 |
|---|---|---|
| Export 1 | 35.14% | 0.3460 |
| Export 2 | 36.04% | 0.3425 |

On a ten class problem, chance is 10 percent. So it had learned something. It had
just not learned nearly enough. Background noise recall collapsed below 9
percent, and the model funnelled clips from nearly every class into gunshot.

To be fair to the tool, this is not really a failure of Teachable Machine. It is
built for quick demonstrations from a handful of examples, and at that it is
genuinely good. It is not built to consume three thousand labelled recordings.
When the data exists and the tool will not take it, the ceiling is the tool.

So we replaced it.

---

## The bug that has killed two models

Before writing a single line of the replacement, we had to deal with something
that had already cost us twice.

A browser model is trained in Python and runs in JavaScript. If the features
those two produce differ even slightly, you get a model that scores beautifully
in training and is quietly useless in the browser, because it is being asked
about numbers it has never seen. We had lost two model generations to exactly
that, and both times it was invisible until the end.

So this time we wrote the feature extraction twice, deliberately, once in Python
and once in JavaScript, and proved they matched **before** training anything.

Same parameters on both sides: 16 kHz audio, a one second window, 1024 sample
frames stepping 512, 64 mel bins from 20 Hz to 8000 Hz, a log, and a per example
mean and standard deviation normalisation.

| Backend | Mean absolute difference | Correlation |
|---|---|---|
| CPU | 3.7e-05 | 1.000000 |
| WebGL | 1.7e-03 | 0.999976 |

We required a mean absolute difference below 0.01 and a correlation above 0.999.
Both passed comfortably, and we checked WebGL separately because that is what a
real browser actually uses and its numerical precision is different.

One small thing worth mentioning, because it is the kind of detail that bites.
16000 samples through a 1024 by 512 window gives 30 frames, not the 31 our
architecture wanted. Rather than fudge it, we zero pad the window to 16384
samples on both sides, which gives exactly 31. Same audio, same answer in both
languages.

---

## The replacement

A small convolutional network. Four blocks of 32, 64, 128 and 128 filters with
batch normalisation and max pooling, global average pooling, dropout, and a
softmax over ten classes. 242,602 parameters, which is a 970 KB download.

Trained on the whole dataset this time. All of it.

| | Accuracy | Macro F1 |
|---|---|---|
| Teachable Machine | 36.04% | 0.3425 |
| **Our browser CNN** | **75.73%** | **0.7555** |

Roughly forty points of accuracy, and more than double the macro F1.

We also checked something people usually skip. Converting a Keras model to
TensorFlow.js can change it. So we scored five clips in Python and again in the
browser through the app's own code path. The largest difference on any class
probability was 0.0002. Then we ran the entire 445 clip test set both ways and
they disagreed on exactly one clip.

---

## The part where we did not get what we wanted

Here is the honest bit.

The browser model still does not meet the requirement.

| Requirement | Result |
|---|---|
| Accuracy at or above 85% | **FAIL** at 75.73% |
| Macro F1 at or above 0.80 | **FAIL** at 0.7555 |
| Gunshot recall at or above 85% | **FAIL** at 71.11% |
| Glass breaking recall at or above 85% | PASS at 86.67% |
| Panic scream recall at or above 85% | **FAIL** at 84.44% |
| Aggression recall at or above 85% | PASS at 95.56% |
| Person asking for help at or above 85% | **FAIL** at 80.00% |

Two of seven. Panic scream misses by half a percentage point, which is a single
clip.

We know exactly where the damage is. Background noise recall is 42 percent.
Twenty six of its forty five clips get scattered into vehicle horn, machinery
fault, gunshot and alarm siren. Because every one of those is a false positive
somewhere else, that single class drags macro F1 down by around 0.03 on its own.
Machinery fault ends up absorbing 70 predictions for 45 actual clips.

The fix is not a cleverer architecture. It is more and more varied background
noise recordings. Background noise is the hardest class to collect because it is
defined by what it is not, and ours is too narrow.

There was one more thing we tried. We wondered whether combining the windows of
a clip differently would help, since taking the maximum across windows means any
one second that faintly resembles an event beats the null class. We compared
three strategies **on the validation split only**, because choosing on the test
split would make the final number meaningless:

| Policy | Validation accuracy | Validation macro F1 |
|---|---|---|
| Max over windows | 80.39% | 0.7978 |
| **Mean over windows** | **82.14%** | **0.8153** |
| Mean of the top 3 windows | 81.48% | 0.8084 |

Mean won on validation, so mean is what shipped. On the test split it came out
0.9 points lower than max would have. That is not a mistake, it is what
selecting honestly looks like. Sometimes the choice you made for the right
reasons does not transfer, and you report it rather than quietly switching back
after seeing the answer.

---

## What we would tell you if you are building something similar

**Prove your features match before you train anything.** If your model trains in
one language and runs in another, that gap will cost you a model at least once.
Make it a gate with a number attached, not a thing you intend to check later.

**Check that your metrics agree with your own confusion matrix.** We found a
metrics file on this project whose headline numbers did not match the matrix
sitting in the same file. Now every number we publish is recomputed from the
matrix and both values are stated. It takes ten lines of code.

**Pick your aggregation on validation and then leave it alone.** The temptation
to peek at the test number and adjust is enormous, and it makes the test number
worthless the moment you give in.

**Know what your tool will not do before you commit to it.** The 100 sample cap
was not hidden. We just did not think about what it meant until we had a
dataset thirty times larger than the tool would accept.

---

## Where it stands

SonicSentinel AI runs end to end. Audio comes in, two independent models score
it, the system compares them and decides whether to alert, send it to a human,
or simply log it. Nothing leaves the machine that recorded it.

The server side model meets every part of the requirement at 90.02 percent. The
browser side model does not, at 75.73 percent, though it is more than twice the
model it replaced.

Half done, in other words, with a clear idea of what the other half needs. That
is a more useful place to be than a number we could not explain.
