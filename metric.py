from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Prediction (left part)
predictions = ['righthand carry milk', 'righthand carry kettle', 'righthand carry cup', 'righthand carry kettle', 'righthand carry milk', 'righthand carry sugar', 'righthand carry sugar', 'righthand carry sugar', 'righthand carry sugar', 'righthand carry kettle', 'righthand carry sugar', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry sugar', 'righthand carry sugar', 'righthand carry kettle', 'righthand carry milk', 'righthand carry sugar', 'righthand carry sugar', 'righthand carry milk', 'righthand carry sugar', 'righthand carry milk', 'righthand carry sugar', 'righthand carry kettle', 'righthand carry milk', 'righthand carry milk', 'righthand carry sugar', 'righthand carry cup', 'righthand carry milk', 'righthand carry cup', 'righthand carry cup', 'righthand carry cup', 'righthand carry sugar', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry kettle', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry sugar', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand close milk bottle', 'righthand take block', 'righthand take egg', 'righthand transfer block to block', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk', 'righthand carry milk']

# Ground truth (right part)
ground_truth = ['righthand take banana', 'righthand take grape', 'righthand transfer pepper to plate', 'righthand transfer carrot to pan', 'righthand transfer strawberry to bowl', 'righthand transfer carrot to pot', 'righthand transfer block to block', 'righthand transfer block to block', 'righthand transfer block to bowl', 'righthand transfer block to bowl', 'righthand transfer block to block', 'righthand transfer block to block', 'righthand transfer block to block', 'righthand transfer block to block', 'righthand push block', 'righthand push block', 'righthand reach banana', 'righthand transfer egg to pressure', 'righthand reach carrot', 'righthand transfer lemon to pan', 'righthand reach block', 'righthand reach block', 'righthand reach block', 'righthand transfer block to block', 'righthand transfer block to block', 'righthand transfer block to bowl', 'righthand take apple', 'righthand take carrot', 'righthand transfer pepper to pan', 'righthand pour kettle into cup', 'righthand pour bottle into cup', 'righthand drop block', 'righthand take eggplant', 'righthand transfer carrot to pot', 'righthand transfer lemon to block', 'righthand transfer block to bowl', 'righthand transfer block to bowl', 'righthand take block', 'righthand take block', 'righthand take block', 'righthand take egg', 'righthand reach egg', 'righthand reach spoon', 'righthand reach block', 'righthand reach block', 'righthand reach egg', 'righthand reach spoon', 'righthand reach block', 'righthand reach block', 'righthand reach apple', 'righthand reach orange', 'righthand reach block', 'righthand reach strawberry', 'righthand take apple', 'righthand take block', 'righthand take block', 'righthand take block', 'righthand take egg', 'righthand transfer block to block', 'righthand transfer block', 'righthand transfer lemon to block', 'righthand transfer knife to block', 'righthand transfer block to banana', 'righthand transfer spoon to block', 'righthand transfer lemon to block']
# Compute BLEU-1 to BLEU-4
smooth = SmoothingFunction().method1
bleu_scores = {1: [], 2: [], 3: [], 4: []}

for pred, gt in zip(predictions, ground_truth):
    gt_tokens = gt.lower().split()
    pred_tokens = pred.lower().split()
    for n in range(1, 5):
        weight = tuple([1.0/n if i < n else 0 for i in range(4)])
        score = sentence_bleu([gt_tokens], pred_tokens, weights=weight, smoothing_function=smooth)
        bleu_scores[n].append(score)

# Compute average BLEU scores
avg_bleu = {f'BLEU-{n}': sum(scores)/len(scores) for n, scores in bleu_scores.items()}
print(avg_bleu)
