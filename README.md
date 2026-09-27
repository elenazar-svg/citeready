# CiteReady, predicting and improving AI citations for web content

## Summary

CiteReady estimates how likely a web page is to be cited by AI answer engines (ChatGPT, Perplexity, Google AI Overviews) and suggests concrete edits that improve both classic SEO and Generative Engine Optimization (GEO).

## Background

More and more people get answers directly from AI assistants instead of clicking on search results. A page can rank well on Google and still never appear in an AI-generated answer, so the site loses visibility without knowing why.

Problems the idea addresses:

* SEO tools measure rankings and clicks, and almost none measure whether a page is cited in AI answers
* GEO advice today is mostly anecdotal and rarely tested on real data
* small businesses, freelancers and agencies cannot afford manual audits of hundreds of pages
* many pages written for keywords are hard for language models to extract, because the answer is buried under long intros or missing structure

Personal motivation. I have worked for over 20 years as an SEO and web consultant in Rome, and in the last year my clients keep asking the same question: "why does ChatGPT mention our competitors and not us?". I want to answer with data instead of opinions.

## How is it used?

The user works on a page and wants to know whether AI engines will use it as a source.

1. The user enters a URL (or opens the page in the WordPress editor) and adds the questions the page should answer. The tool can also suggest questions from Google Search Console queries.
2. CiteReady splits the page into passages and compares each passage with each question.
3. A model returns a **citation score from 0 to 100** for every question and highlights the passage most likely to be quoted.
4. The tool explains the score with readable signals, such as "the answer appears after 300 words", "no author or update date", "missing FAQ or structured data", "key entities not defined".
5. On request, an AI model drafts a rewrite of the weak passage. The editor reviews and approves every change before publishing.

Users are SEO consultants, web agencies, editors and e-commerce owners. They need clear explanations more than technical metrics, support for languages other than English (Italian first), and full control over what gets published.

A simplified version of the scoring model:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# One row per passage:
# [similarity to question, answer in first 50 words, has FAQ schema,
#  length in hundreds of words, has author and date]
X = np.array([
    [0.91, 1, 1, 2.1, 1],
    [0.45, 0, 0, 6.5, 0],
    [0.83, 1, 0, 3.0, 1],
    [0.52, 0, 1, 4.8, 0],
    [0.88, 1, 1, 1.7, 0],
    [0.38, 0, 0, 7.2, 1],
])
y = np.array([1, 0, 1, 0, 1, 0])  # 1 = cited by an AI answer engine

model = LogisticRegression().fit(X, y)

new_passage = np.array([[0.80, 0, 0, 5.5, 1]])
prob = model.predict_proba(new_passage)[0, 1]
print("Citation probability %.0f%%" % (100 * prob))
```

## Data sources and AI methods

I collect the main dataset myself.

* A list of real questions per sector, taken from Search Console queries, "People also ask" boxes and questions that clients receive from customers.
* The questions are sent at regular intervals to AI answer engines through their official APIs where available, and the cited URLs are recorded. A cited page gets label 1.
* For the same questions, pages that rank in the top 10 on Google and are not cited get label 0.
* For every page the tool extracts structure, schema.org markup, entities, freshness and authorship signals.

| Method | What it is used for |
| ----------- | ----------- |
| Text embeddings (e.g. sentence-transformers) | Semantic similarity between a question and each passage |
| Logistic regression, then gradient boosting | Predicting citation probability with interpretable features |
| Nearest neighbors | Showing cited pages similar to the user's page as examples |
| Named entity recognition | Comparing entity coverage with cited competitors |
| Large language model (via API) | Drafting rewrites and FAQ suggestions, always reviewed by a human |

## Challenges

* **The score is a probability and gives no guarantee.** AI engines are black boxes and change often, so the model needs regular retraining and the tool must say so clearly.
* The data shows correlation. A signal linked to citations does not necessarily cause them, and only controlled tests on real pages can confirm causality.
* The dataset may favour English content and large brands, which would make scores less reliable for small Italian sites.
* Ethics. A tool like this could be used to produce manipulative or uniform content. CiteReady should reward accuracy, clarity and verified sources, and AI rewrites can introduce errors, so human approval stays mandatory.
* Data collection must respect the terms of service of each engine and robots.txt, and the tool does not process personal data.

## What next?

* A WordPress plugin that shows the score directly in the editor
* A monitoring dashboard that tracks citations over time for a set of pages
* A/B tests on real client pages to measure which changes actually increase citations
* Support for more languages and more answer engines

To move forward I need help from a data scientist for model validation, a WordPress developer for the plugin, and a few agencies willing to join a pilot with their pages.

## Acknowledgments

* Aggarwal et al., [GEO: Generative Engine Optimization](https://arxiv.org/abs/2311.09735), the research paper that inspired the idea
* [schema.org](https://schema.org) vocabulary for structured data
* Open source libraries [scikit-learn](https://scikit-learn.org) and [sentence-transformers](https://www.sbert.net)
