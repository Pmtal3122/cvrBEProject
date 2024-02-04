from sklearn.metrics.pairwise import cosine_similarity
import operator
    
def recommend(word, data, word_vectors):
    vals = list(data.values())
    # print("vals:")
    # print(vals)
    maxAvg=0
    maxInd = -1
    ind = 0
    indices = dict()
    for val in vals:
        avg = 0
        vals_length = 0
        print(val)
        for value in val.items():
            try:
                avg = avg + (value[1] * cosine_similarity([word_vectors[word]], [word_vectors[value[0]]])[0])
                vals_length = vals_length + value[1]
                # indices.update({cosine_similarity([word_vectors[word]], [word_vectors[value]])[0], ind})
            except:
                continue
        avg = avg / vals_length
        indices[ind] = avg
        if avg > maxAvg:
            maxInd = ind
            maxAvg = avg
        ind = ind + 1
    print("The indices dict is as follows")
    print(indices)
    sorted_indices = dict(sorted(indices.items(), key=operator.itemgetter(1), reverse=True))
    print("The sorted indices dict is as follows")
    print(sorted_indices)
    
    indices = {key: sorted_indices[key] for key in list(sorted_indices)[:3]}
    print("The top 3 indices are")
    print(indices)
    
    
    return indices
    # return maxInd

def recommend_categories(word, data, word_vectors):
    keys = list(data.keys())
    indices = dict()
    ind = 0
    for key in keys:
        indices[ind] = cosine_similarity([word_vectors[key.lower()]], [word_vectors[word]])
    
    indices = dict(sorted(indices.items(), key=operator.itemgetter(1), reverse=True))
    indices = {key: indices[key] for key in list(indices)[:3]}
    
    return indices