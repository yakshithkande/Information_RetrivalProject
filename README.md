# Information_RetrivalProject
!pip install -q rank-bm25

import re
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

plt.rcParams["figure.figsize"] = (6,4)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

EN_STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
LEGAL_STOPWORDS = {"petitioner","respondent","appellant","defendant","plaintiff","court","judge","bench","hon","honourable","learned","state","india","case","appeal","order","petition","section","act","article","clause"}
ALL_STOPWORDS = EN_STOPWORDS.union(LEGAL_STOPWORDS)

def strip_html(x): return re.sub(r'<[^>]+>', ' ', str(x))
def remove_legal_citations(t):
    t = re.sub(r'\[[^]]+\]', ' ', t)
    t = re.sub(r'\(\d{4}\)\s*\d*\s*[A-Za-z]+\s*\d*', ' ', t)
    t = re.sub(r'\bAIR\s+\d{4}\s+[A-Za-z]+\s+\d+\b', ' ', t)
    t = re.sub(r'\b\d+\s*SCC\s*\d+\b', ' ', t)
    return t
def remove_sections(t):
    t = re.sub(r'\bsection\s+\d+[A-Za-z]*\b', ' ', t)
    t = re.sub(r'\bu/s\.?\s*\d+[A-Za-z]*\b', ' ', t)
    t = re.sub(r'\bunder\s+section\s+\d+[A-Za-z]*\b', ' ', t)
    return t
def clean_text_basic(t):
    t = strip_html(t).lower()
    t = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', t)
    t = remove_legal_citations(t)
    t = remove_sections(t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()
def preprocess_tokens(t):
    t = clean_text_basic(t)
    toks = nltk.word_tokenize(t)
    out = []
    for tok in toks:
        if tok not in ALL_STOPWORDS and len(tok) >= 3:
            out.append(LEMMATIZER.lemmatize(tok))
    return out
def preprocess_text(t): return " ".join(preprocess_tokens(t))

DATA_URL = "https://raw.githubusercontent.com/NoelShallum/Indian_SC_Judgment_database/main/final_judge_database.csv"
df = pd.read_csv(DATA_URL)
df.columns = df.columns.str.strip()

def norm(s): return re.sub(r'[^a-z]', '', s.lower())
cols = {c: norm(c) for c in df.columns}
def find_col(keyword):
    for c,n in cols.items():
        if keyword in n:
            return c
    raise ValueError(keyword)

ISSUES_COL = find_col("issues")
TITLE_COL = find_col("casetitle")
DATE_COL = find_col("date")
CITED_COL = None
for key in ["citedcases","cited","citation"]:
    try:
        CITED_COL = find_col(key)
        break
    except:
        pass

df = df.dropna(subset=[ISSUES_COL]).copy()
df.reset_index(drop=True, inplace=True)
df["case_id"] = df.index
df["year"] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.year

if CITED_COL is not None:
    combined = df[TITLE_COL].astype(str)+" "+df[ISSUES_COL].astype(str)+" "+df[CITED_COL].astype(str)
else:
    combined = df[TITLE_COL].astype(str)+" "+df[ISSUES_COL].astype(str)

df["tokens"] = combined.apply(preprocess_tokens)
df["clean_text"] = df["tokens"].apply(lambda x:" ".join(x))

corpus_tokens = df["tokens"].tolist()
bm25 = BM25Okapi(corpus_tokens)
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=5, ngram_range=(1,2))
X_tfidf = tfidf_vectorizer.fit_transform(df["clean_text"])
svd = TruncatedSVD(n_components=200, random_state=42)
X_lsa = svd.fit_transform(X_tfidf)
X_lsa_norm = X_lsa/(np.linalg.norm(X_lsa, axis=1, keepdims=True)+1e-9)

k=6
kmeans=KMeans(n_clusters=k,random_state=42,n_init=10)
df["cluster"]=kmeans.fit_predict(X_lsa_norm)

def min_max_normalize(a):
    mi,ma=np.min(a),np.max(a)
    return np.zeros_like(a) if ma-mi==0 else (a-mi)/(ma-mi)

def parse_year_range(s):
    if "-" not in s: return None, None
    a,b=s.split("-")
    return int(a), int(b)

def hybrid_search(query, year_from=None, year_to=None, top=10):
    q_tokens=preprocess_tokens(query)
    q_clean=" ".join(q_tokens)
    mask=np.ones(len(df),bool)
    if year_from is not None: mask &= df["year"].fillna(0)>=year_from
    if year_to is not None: mask &= df["year"].fillna(0)<=year_to
    idx=np.where(mask)[0]
    if len(idx)==0: idx=np.arange(len(df))

    bm25_scores=np.array(bm25.get_scores(q_tokens))[idx]
    cand_local=np.argsort(-bm25_scores)[:80]
    cand_idx=idx[cand_local]
    b=bm25_scores[cand_local]

    q_vec=tfidf_vectorizer.transform([q_clean])
    t=cosine_similarity(q_vec,X_tfidf[cand_idx]).flatten()
    q_lsa=svd.transform(q_vec)
    q_lsa_norm=q_lsa/(np.linalg.norm(q_lsa,axis=1,keepdims=True)+1e-9)
    l=(X_lsa_norm[cand_idx]@q_lsa_norm.T).flatten()

    bn=min_max_normalize(b); tn=min_max_normalize(t); ln=min_max_normalize(l)
    final=0.4*bn+0.3*tn+0.3*ln
    top_local=np.argsort(-final)[:top]
    out_idx=cand_idx[top_local]

    r=df.iloc[out_idx].copy()
    r["final_score"]=final[top_local]

    qset=set(q_tokens)
    matches=[]
    for i in out_idx:
        doc=df.loc[i,"tokens"]
        mm=[tok for tok in doc if tok in qset][:5]
        matches.append(", ".join(mm))
    r["match_terms"]=matches
    return r

def similar_cases(case_id, top=5):
    v=X_lsa_norm[case_id]
    scores=X_lsa_norm@v
    scores[case_id]=-1
    top_idx=np.argsort(-scores)[:top]
    r=df.iloc[top_idx].copy()
    r["similarity"]=scores[top_idx]
    return r

def print_results(r):
    for i,row in r.reset_index(drop=True).iterrows():
        print(f"\nRank {i+1} | CaseID={row['case_id']} | Score={row['final_score']:.4f}")
        print("Title:",row[TITLE_COL])
        print("Year:",row["year"]," | Cluster:",row["cluster"])
        print("Match terms:",row["match_terms"])
        print("Issues:",str(row[ISSUES_COL])[:260],"...")

while True:
    q=input("\nEnter your legal query (or 'exit'): ").strip()
    if q.lower() in ["exit","quit"]: break

  yr = input("Enter year range (ex: 1990-2005) or blank: ").strip()
    yf,yt = parse_year_range(yr) if yr else (None,None)

  r = hybrid_search(q, year_from=yf, year_to=yt, top=10)
    print_results(r)
    sel=input("\nEnter rank number (1–10) for similar cases or blank: ").strip()
    if sel.isdigit():
        cid=r.iloc[int(sel)-1]["case_id"]
        sims=similar_cases(cid)
        for i,row in sims.reset_index(drop=True).iterrows():
            print(f"\nSimilar {i+1} | CaseID={row['case_id']} | Sim={row['similarity']:.4f}")
            print("Title:",row[TITLE_COL])
            print("Year:",row["year"],)
            print("Issues:",str(row[ISSUES_COL])[:260],"...")
