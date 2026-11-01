# Phishing Website Detection: Machine Learning Assignment 2

**Live app:** <STREAMLIT_APP_LINK>
**GitHub repo:** <GITHUB_REPO_LINK>

## a. Problem statement
Classify whether a website URL is a legitimate page or a phishing attempt designed to steal user
credentials. Blacklists cannot keep up with newly created URLs, so five classification models are
trained on 30 structural URL and page attributes and compared on unseen test data using Accuracy,
AUC, Precision, Recall, F1 and MCC. This is a binary classification problem
(`class` = 1 phishing, 0 legitimate).

## b. Dataset description
- **Source:** UCI Machine Learning Repository, "Phishing Websites" dataset (Mohammad, Thabtah and
  McCluskey, 2012). <ADD_EXACT_DATASET_URL>
- **Raw size:** 11,054 rows, 30 features, 1 target. After removing 5,205 duplicate rows (and the
  `Index` column) 5,849 unique rows remain.
- **Split:** stratified 75/25, random seed 2025 -> 4,386 training rows (`model/train_data.csv`) and
  1,463 test rows (`test_data.csv`). No row appears in both files.
- **Target:** `class` (1 = phishing, 0 = legitimate; recoded from the original -1 / 1).
- **Feature values:** every feature is already encoded as `-1`, `0` or `1`.
  Generally `1` = legitimate sign, `0` = suspicious, `-1` = phishing sign (for example `HTTPS = -1`
  means no valid HTTPS).

### Column meanings
| Column | Meaning |
|---|---|
| UsingIP | URL uses an IP address instead of a domain name |
| LongURL | Length of the URL (very long URLs hide the real domain) |
| ShortURL | URL was shortened with a service such as bit.ly or tinyurl |
| Symbol@ | URL contains an `@` symbol (browser ignores text before it) |
| Redirecting// | `//` appears in the URL after the protocol, indicating a redirect |
| PrefixSuffix- | Domain name contains a `-` (phishers imitate brands, e.g. paypal-secure.com) |
| SubDomains | Number of dots/subdomains in the domain part |
| HTTPS | Uses HTTPS with a trusted certificate issuer and adequate age |
| DomainRegLen | Domain registration period (short registrations are suspicious) |
| Favicon | Favicon is loaded from an external domain |
| NonStdPort | Uses a non-standard port |
| HTTPSDomainURL | The word "https" is used inside the domain name to mislead users |
| RequestURL | Share of page objects (images, videos) loaded from another domain |
| AnchorURL | Share of `<a>` links pointing to another domain or to no real page |
| LinksInScriptTags | Share of `<meta>`, `<script>` and `<link>` tags pointing to other domains |
| ServerFormHandler | Form handler is empty, `about:blank` or points to another domain |
| InfoEmail | Page submits user information to an email address (`mailto:`) |
| AbnormalURL | Host name is not part of the URL as recorded in WHOIS |
| WebsiteForwarding | Number of times the page is redirected |
| StatusBarCust | JavaScript changes the status bar to hide the real link |
| DisableRightClick | Right-click is disabled by script |
| UsingPopupWindow | Page opens pop-up windows that ask for text input |
| IframeRedirection | Page uses invisible iframes to load other content |
| AgeofDomain | Age of the domain (newly created domains are suspicious) |
| DNSRecording | Domain has a DNS record |
| WebsiteTraffic | Website popularity ranking (e.g. Alexa) |
| PageRank | Google PageRank of the page |
| GoogleIndex | Page is indexed by Google |
| LinksPointingToPage | Number of other sites linking to this page |
| StatsReport | Host/IP appears in phishing statistics reports (e.g. PhishTank) |
| **class** | **Target: 1 = phishing, 0 = legitimate** |

## c. Models and evaluation metrics
Models: Logistic Regression, Decision Tree, kNN, Naive Bayes (Gaussian), SVM (RBF kernel).
Logistic Regression, kNN and SVM use a StandardScaler inside a scikit-learn pipeline. All models were
evaluated on `test_data.csv` (1,463 rows).

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9139 | 0.9756 | 0.9199 | 0.9126 | 0.9162 | 0.8276 |
| Decision Tree | 0.9344 | 0.9660 | 0.9252 | 0.9497 | 0.9373 | 0.8688 |
| kNN | 0.9125 | 0.9748 | 0.9109 | 0.9205 | 0.9157 | 0.8248 |
| Naive Bayes | 0.6740 | 0.9680 | 0.6128 | 1.0000 | 0.7599 | 0.4472 |
| SVM | 0.9446 | 0.9846 | 0.9542 | 0.9377 | 0.9459 | 0.8894 |

> Replace these numbers with the output of your own run on the BITS Virtual Lab.

### Observations
| ML Model | Observation |
|---|---|
| Logistic Regression | Good, stable baseline. Precision and recall are close, but a linear boundary misses some non-linear patterns. |
| Decision Tree | Best recall (0.9497), so it catches the most phishing sites, but a lower AUC shows its probability ranking is less refined. |
| kNN | Similar to Logistic Regression. Slightly lower scores, and prediction is slower because every training row is compared. |
| Naive Bayes | Recall is 1.00 but precision is only 0.61: it flags almost everything as phishing. The Gaussian assumption does not suit discrete -1/0/1 features. |
| SVM | Best Accuracy, AUC, Precision, F1 and MCC. The RBF kernel captures non-linear interactions between features. |

**Overall best model: SVM** (highest MCC, 0.8894, narrowly ahead of Decision Tree at 0.8688). Models are
ranked by MCC, with F1 as the tie-breaker, because MCC uses all four confusion-matrix counts.

## d. Project structure
```
app.py             Streamlit app (upload test CSV, evaluate all models)
requirements.txt   Python dependencies
README.md          This file
test_data.csv      Test data (upload this in the app)
model/             train_pipeline.py, train_data.csv, saved models (*.pkl), metrics_summary.csv
```

## e. How to run
```bash
pip install -r requirements.txt
python model/train_pipeline.py     # trains and saves the 5 models
streamlit run app.py
```
Upload `test_data.csv` in the app and click **Evaluate all models**.
