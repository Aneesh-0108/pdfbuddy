loading -> chunking -> indexing -> embedding -> FAISS -> Gemini

threshold while testing for retreival of FAISS :

| Score Range | Meaning             |
| ----------- | ------------------- |
| 0.0 – 0.8   | Very relevant       |
| 0.8 – 1.2   | Moderately relevant |
| 1.2 – 1.5   | Weak relevance      |
| > 1.5       | Probably unrelated  |


The retrieval system was evaluated using both in-domain and out-of-domain queries. In-domain questions such as "What is booting?" produced significantly lower similarity scores (0.73), indicating strong semantic relevance. Out-of-domain questions such as "What is machine learning?" produced higher scores (~1.6), demonstrating that the retriever correctly identifies the absence of relevant information in the document.