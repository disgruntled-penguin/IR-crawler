# SpinTrace demo video script

Speakers: Pustak does Parts 1, 2, most of 7, and 8. Nishita does the app intro, Parts 3 to 6, plus in Part 7 the judging clips and their numbers.
Length: about 11:10 of talking at a normal pace, plus pauses. The PDF allows 5 to 8 minutes, so cut about 3:20 before recording.
Machine: everything runs on Nishita's laptop. Record each part separately and join them in editing.

[LAPTOP] marks what to do on screen. Quoted text is what to say.

## Before recording (not filmed)

1. Terminal: very large font (Cmd + several times), dark theme, then `cd ~/Desktop/IR-crawler`.
2. App: open a second terminal tab and run `python3 -m spintrace app`. Open http://127.0.0.1:8766 in the browser and zoom to about 150%.
3. Warm up the app: Trace tab, paste `36261800e15f8b91`, press Trace and wait about 20 s. After that every trace is instant.
4. Browser tabs, in this order:
   - NewsGuard: https://www.newsguardtech.com/misinformation-monitor/august-2023
   - Farm copy: https://web.archive.org/web/20230809202538/https://www.globalvillagespace.com/GVS-Sports/darren-waller-pursues-music-alongside-nfl-success/
   - Original: https://web.archive.org/web/20230507093623/https://www.nytimes.com/2023/05/07/sports/football/darren-waller-giants-music.html
   - The SpinTrace app
5. Judging clips (d/n and r/x) loaded in the video editor.
6. Only one crawler at a time. If a crawl is already running, a new one refuses to start.

## Part 1. The problem: Pustak

[LAPTOP] NewsGuard tab.

> "Hi, we're Pustak and Nishita, and this is SpinTrace, our project for Track 4: web crawling and web integrity.
>
> There's a growing problem online: AI content farms. These are websites that take real news articles, have a chatbot reword them, and publish them as their own. NewsGuard, an organisation that tracks misinformation, has found more than 3,700 of these sites. In one test, they gave 43 AI-rewritten articles to a well-known plagiarism checker. It couldn't find the original for 34 of them. That's 79 percent."

[LAPTOP] Farm copy and New York Times side by side.

> "Here's a real case from their report. On the left, a farm site's article about the NFL player Darren Waller. On the right, the New York Times story it was taken from. The newspaper that did the reporting loses the credit and the readers. Crawlers and search engines do check for duplicates, but they look for repeated runs of words. Reword the article, and the check is fooled. Nishita will show you what we built."

## App intro: Nishita

[LAPTOP] App tab, Search tab, with both boxes ticked. Type `bangladesh measles` and press Search.

> "This is SpinTrace, the app we built over our crawl. It has three parts. Search lets you search all the news we crawled, like a small search engine. Here's a search for the measles outbreak in Bangladesh. First is the original article, from The Conversation. Down here at number eight is scroll.in's version, with a 'copy' badge naming where it came from. The Conversation allows other outlets to republish its articles, so that's fine, but readers should still find the original first, and this box makes sure originals rank above their copies."

[LAPTOP] Press Trace on the scroll.in result. Scroll slowly through the page.

> "Every result has a Trace button. Trace shows how SpinTrace decided it was a copy: it searched with the article's rarest words, like Gavi, Seychelles and rubella, found The Conversation's article, published two weeks earlier, and matched the two articles sentence by sentence. Verdict: derived. We'll go through these steps properly in a moment."

[LAPTOP] Click Site integrity. Point at the devdiscourse.com row.

> "And Site integrity shows, for every news site we crawled, how many of its articles were traced to another outlet, and to which ones. Devdiscourse, for example: 20 of its 692 articles, mostly from indiatimes.com. Pustak will explain the idea behind how it works."

## Part 2. The idea: Pustak

[LAPTOP] Back to the two Waller articles side by side. Highlight a shared name or number in both as you say it, for example "Interscope" or "21 other N.F.L. players".

> "Our idea is simple. A chatbot can change every word, but it can't change the facts. The names, numbers, dates and quotes stay the same. Look: the same people, the same record label, the same numbers on both sides. In search terms these are rare words, and rare words act like a fingerprint.
>
> SpinTrace works in four steps. A crawler collects news from real websites. Everything goes into a search index. For any article, we search with its rarest words to find where it came from. Then we check the best matches sentence by sentence. Classic search does the finding, and AI only does the final check. Nishita will walk you through it."

## Part 3. The crawler: Nishita

[LAPTOP] Terminal. Run `python3 -m spintrace crawl --hours 0.05`. Wait silently until FETCH lines are scrolling, then start talking. Cut the wait in editing.

> "This is our crawler running live. It starts from 36 news sources, mostly their RSS feeds. It's polite: before visiting any site, it reads the site's robots.txt, the file where a site says what crawlers may access."

[LAPTOP] Point at a FETCH line and its `gap=` value.

> "This number is the time since our last visit to the same site. We always wait at least 5 seconds, so we never overload anyone. Each article it saves goes into our database with its publish date. The queue of pages to visit and these waiting rules are code we wrote ourselves."

[LAPTOP] Press Ctrl+C. Run `python3 -m spintrace crawl-report` and stay on the top block.

> "Over the full crawl, about 13 hours, we saved more than 32 thousand articles from 118 websites. Pages fetched that robots.txt didn't allow: zero. The report also flagged visits that came too close together, mostly from accidentally running two crawlers at once, so now only one can run at a time."

## Part 4. Catching a rewrite: Nishita

[LAPTOP] App, Trace tab. Paste `36261800e15f8b91` and press Trace. Scroll slowly through the steps. At step three, point at the bold p column, wikinews.org row (0.987).

> "Now the main part: catching a rewrite. This article is an AI summary of a news story about South Korea's president. We hid it among our 21 thousand articles as a test. The addresses ending in .example are our own test rewrites.
>
> Step one: SpinTrace takes the article's 30 highest-idf words: Yoon, Suk, Yeol, impeach, and 190, the vote count. Rewording changes phrases, but these facts survive. This is index elimination: we only read the postings lists of rare terms, so 21 thousand articles narrow down to a handful of suspects. Any direct quotes also become phrase queries on our positional index. And a parametric filter on publish date allows only earlier articles, because a source has to come first.
>
> Step two: the candidates are ranked by tf-idf cosine similarity, with lnc.ltc weighting, and the original Wikinews story is among the top matches.
>
> Step three: the top five get a closer check. Several signals, like sentence coverage, sentence order, and shared names and numbers, go into a logistic model trained on our dev split, which gives one probability. For the Wikinews original it's 0.987, well above our threshold of 0.804. Every result we report is measured on a separate test split, so nothing is tuned on the answers.
>
> Step four is the proof. Sentence embeddings compare sentences by meaning, not exact words. 'Abstained from voting' matches 'did not participate'. Every sentence of the rewrite matches the original, in the same order. Verdict: derived from the Wikinews article, published 20 hours earlier."

[LAPTOP] Terminal. Run `python3 -m spintrace pair 36261800e15f8b91 af2e0c39bf2263f9`

> "Now here's why this matters. This is the standard duplicate check crawlers use. We built it into SpinTrace as our starting point. It splits both articles into five-word shingles and measures their Jaccard overlap, estimated with MinHash. On the same pair: a Jaccard of only 0.05, and MinHash with LSH doesn't even make it a candidate. The standard check would have said 'not a copy' and let it through. This check is one of six baselines we compare against; the others are exact hashing, shingle containment, tf-idf cosine, BM25, and dense retrieval with embeddings. Pustak will show how they do."

[LAPTOP] App, Trace tab. Paste `c32b00317dd1b362` and press Trace.

> "And on the real case from NewsGuard: out of 21 thousand articles, SpinTrace ranks the New York Times first, so precision at 1, and traces the farm's copy straight back to it."

## Part 5. Search: Nishita

[LAPTOP] App, Search tab. Untick "rank originals above copies". Search `king sunny ade`.

> "This matters for search too. Here's a search over our crawl for the Nigerian musician King Sunny Adé. Ranked by relevance alone, the top result is scroll.in's republished version of an article from The Conversation. The original comes second."

[LAPTOP] Tick the box.

> "Turn on our originality score, the static quality g of d. The net score is relevance plus lambda times g of d, so copies score lower. Now the original is first, and the copy drops below it."

[LAPTOP] Click Site integrity for about 5 seconds.

> "We can also see, for each news site, what share of its articles were traced to another outlet."

## Part 6. A limitation: Nishita

[LAPTOP] Trace tab. Paste `27d21f82493f53bb` and press Trace.

> "It doesn't always get it right. We asked an AI to write an article using only the facts of a cricket story, the names, the score and one quote, without ever seeing the original. That's not copying. But because the facts and the quote match, SpinTrace still calls it derived. In IR terms, this is a false positive, and false positives are what cost us precision. Telling 'same story' apart from 'copied' is our hardest problem. It happens about 16 percent of the time. Over to Pustak for the results."

## Part 7. Results

### Pustak: the test results

[LAPTOP] Terminal. Run `open results/detection_f1_by_level.png`

> "So how well does it work overall? We made test rewrites at six levels of difficulty. On the left are exact copies and light edits. On the right are what AI farms actually do: SEO rewrites, summaries and translations. We compared SpinTrace with six standard methods, and each one got its best settings, so the comparison is fair.
>
> This chart scores how well each method spots a rewrite; higher is better. The green line is the standard duplicate check from the terminal: perfect on exact copies, then it collapses. It's 0.32 on synonym swaps and almost zero on AI rewrites. The blue line is SpinTrace: it stays between 0.78 and 0.96 at every level."

[LAPTOP] Run `open results/p_at_1_by_level.png`

> "This chart asks a different question: when we search for the source, is the right original ranked first? The standard check collapses again. But the other search-based methods are almost as good as us here. That's exactly why we use search to find sources. Finding is the easy part. The hard part is deciding whether an article is actually a copy, and that's where SpinTrace pulls ahead on the first chart."

[LAPTOP] Run `open results/index_elimination.png`

> "Using only rare words is also fast. With just the 10 rarest words of an article, we find its source first 97 percent of the time, while reading only a tiny part of the index. The 10 most common words read 170 times more and find it only 71 percent of the time. Rare facts are the fingerprint."

[LAPTOP] Stay on the chart, or show the app's Search tab.

> "In search, ranking originals above copies works: the original comes first for 53 percent of our test searches, up from 10 percent. And on real NewsGuard cases, SpinTrace traced GlobalVillageSpace back to the New York Times. A second case, a reshuffled shopping list from TopGolf.kr, it found but couldn't confirm. Nishita will tell you how we checked it by hand."

### Nishita: human judging

[LAPTOP] Play the d/n judging clip, sped up.

> "Test data isn't enough, so we also checked real results by hand. SpinTrace flagged 174 pairs of articles in our live crawl. Pustak and I each judged every pair separately, as copied or not, on a page that hides SpinTrace's answer so it can't influence us."

[LAPTOP] Play the r/x judging clip, sped up.

> "We did the same for search: for 15 queries on our live crawl, we marked each of the top 10 results relevant or not."

[LAPTOP] Terminal. Run `python3 -m spintrace judge live --metrics | head -13` and point at `precision_of_flags: 0.845`.

> "Counting a pair as a copy if either of us marked it, 147 of the 174 flags were real copies: a precision of 85 percent."

[LAPTOP] Run `column -s, -t < results/search_eval_judged.csv | head -3` and point at p@10, 0.9333.

> "And in search, precision at 10 was 0.93: 93 percent of the top-10 results were relevant, by the same rule."

### Pustak: where we lose

> "Where we lose: on exact copies, the standard check is slightly better and much cheaper, so a real crawler should run both. And articles written independently from the same facts still fool us 16 percent of the time, as Nishita showed."

## Part 8. Close: Pustak

[LAPTOP] Back to the app, or the two Waller articles side by side.

> "So, SpinTrace takes the duplicate check every crawler already has and teaches it to catch paraphrases, using the facts a rewrite can't change. And it ranks originals above their copies. Next, we want the crawler to visit suspected copy sites more often, and to catch English news rewritten into Hindi. All our code is on GitHub, and everything you saw ran live. Thanks for watching."

## After recording (not filmed)

Restore the files that `crawl-report` rewrites. Run this after the last take, even if you ran the report only once:

```
git restore README.md results/summary.md results/crawl_report.json
```

Then stop the app with Ctrl+C in its terminal tab.

## Cutting to under 8 minutes: suggested order

1. Part 5: the Site integrity glimpse (the app intro already shows it).
2. Part 7: the speed chart (`index_elimination.png`).
3. Part 2: the "Look: the same people..." sentence.
