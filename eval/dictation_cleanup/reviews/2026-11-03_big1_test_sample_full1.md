# Hand check: full1 on the big1 test set (150 random cases)

Run: `runs/2026-11-03_full1-q8_0-greedy_cleanup_t1_short_big1_test` (llama.cpp Q8_0, greedy, 5080). Sample: 150 cases
drawn at random (seed 150) from the 2,500. Judged by Claude with Khang's rule (meaning kept: the main ideas, as
complete, polite sentences; nothing invented, alternatives included). Full pass = both human checks and the
automatic checks.

| | Count |
| --- | --- |
| Automatic pass in the sample | 143 / 150 (95%) |
| Full pass by Khang's rule | **132 / 150 (88%)** |
| Whole test set, automatic | 2,389 / 2,500 (95.6%) |

With 150 cases the 88% is good to about ±5 points. The automatic checks run about 7 points high: they check that
names, numbers and terms survive, not what the sentence means.

## The 18 failures

**1. Word list: other list words offered as alternatives, or the wrong one picked (7).** In every one the word list
held the right term and two other words of the same field.
- Alternatives invented from the list: B1201 (quinoa, "or cumin"), B0231 (takeout, "or gluten-free takeout |
  vegetarian takeout"), B1182 (quarterback, "or power soccer player"), B1867 (graduation, "or anniversary |
  potluck"), B0450 (reservation, "or wheelchair accessible reservation"). The main reading was right each time.
- Wrong pick: B0058 "s- score server" became "Minecraft speedrun server" (meant Discord); B2422 "silly bus" became
  "bus" with syllabus on the list.

**2. Misheard term with only a weak hint: a wrong guess, or the mishearing kept (5).** B0376 "hen controls" became
"brake controls" (meant hand controls); B0742 "a foe" became "axe" (AFO); B1891 "access a bull unit" became
"access to my unit"; B0618 "guest lift" and B1024 "recipe" (receipt) were kept as heard.

**3. Small words dropped: the sentence changed meaning (6).** B2009 "buy new shoes, Noah" (for Noah); B0362
"I'm switching to sertraline" (meant "Can I switch the sertraline to the evening?"); B1871 a question became a
statement; B1560 "I will send the paper back" (meant "The paper is back"); B1456 "I have a pain medication now"
(meant "Can I have my pain medication now?"); B2203 "update my address" (meant "my will").
Two of these are unfair cases: the noise dropped "will" (a noun here) in B2203 and "can" in B1456, so the heard text
no longer holds what was meant. Fix in the next test set: never drop "can", "will" or "do" at the start of a
question, or "will" after "my".

## What to look for in big1

Pattern 1 should improve: big1's training has word lists with three field words and answers without alternatives.
Patterns 2 and 3 are the real gaps: when a term has no clear evidence, the safer answer keeps the heard words and
marks them unsure instead of guessing.
