1. Get both new APIs (track audio features and track audio analysis to sync up into one single json for n songs)
2. Create functions to generate embeddings of the given data. We split the data into different vectors:
        - Trivial/Metadata features
        - Lyrical features
        - Audio-visual features (CNN based extraction of features from audio files) **Speed this up**
3. Start a milvus db and generate schema for the collection.
4. Send data to collection.
5. Run tests and perform error analysis.
6. Generate new features if needed.


Estimated time for project completion : 1 Day (legit)