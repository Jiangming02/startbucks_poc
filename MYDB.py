import faiss
import numpy as np


class FDB:

    def __init__(self, vector_len, use_gpu=False, gpu_id=0):
        db = faiss.IndexFlatL2(vector_len)
        if (use_gpu):
            res = faiss.StandardGpuResources()
            db = faiss.index_cpu_to_gpu(res, gpu_id, db)
        self.db = db
        self.keys = []
        self.vectors = []

    def add(self, vectors, keys):
        for i in range(0, len(keys)):
            self.keys.append(keys[i])
            self.vectors.append(vectors[i])
        self.db.add(np.array(vectors,dtype=np.float32))

    def search(self, vectors, top_k=3):
        D, I = self.db.search(vectors, top_k)
        result = []
        for i in range(0, len(D)):
            res = []
            for j in range(0, len(D[i])):
                if(I[i][j]<0 or I[i][j]>=len(self.keys)):
                    continue
                score = 1.0 /( 1+D[i][j])
                # if(score<0):
                #     score = 0
                res.append({"key": self.keys[I[i][j]], "score": score})
            result.append(res)
        return result

    def reset(self):
        self.db.reset()
        del self.keys
        del self.vectors
        self.keys = []
        self.vectors = []

class MyDB:
    def __init__(self, vector_len, use_gpu=False, gpu_id=0):
        self.vector_length = vector_len
        if (use_gpu):
            self.FDB = FDB(self.vector_length, use_gpu=True, gpu_id=gpu_id)
        else:
            self.FDB = FDB(self.vector_length)

    def reset(self):
        self.FDB.reset()

    def add_texts(self, texts, vectors):
        self.FDB.add(vectors, texts)
        return vectors

    def search(self, vectors, top_k=3, min_score=0.5):
        result1 = self.FDB.search(np.array(vectors,dtype=np.float32), top_k=top_k)
        results = []
        for i in range(0, len(result1)):
            arr = []
            for obj in result1[i]:
                obj['score'] = float(obj['score'])
                if (obj['score'] >= min_score):
                    arr.append(obj)
            results.append(arr)
        return results


if (__name__ == '__main__'):
    import asyncio
    from tools.GetTextEmbeddings import get_embeddings
    v = len(asyncio.run(get_embeddings(['你好']))[0])
    print('Vector Length',v)
    db = MyDB(v, use_gpu=False,gpu_id=4)
    db.add_texts(['胸闷', '发热', '咳嗽'], asyncio.run(get_embeddings(['胸闷', '发热', '咳嗽'])))
    vectors_serch = asyncio.run(get_embeddings(["有一点胸闷", "体温39摄氏度"]))
    print(db.search(vectors_serch,min_score=0))
    db.reset()
    print(db.search(vectors_serch, min_score=0))
    print('---DONE---')
