import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from models.reward_model import RewardModel
from utils.cot_sampler import CoTSampler
from utils.perplexity_scorer import PerplexityScorer
import argparse

class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['prompt'], item['chosen'], item['rejected'], item.get('weight', 1.0)

def collate_fn(batch, tokenizer, max_len):
    prompts, chosens, rejecteds, weights = zip(*batch)
    chosen_texts = [p + " " + c for p, c in zip(prompts, chosens)]
    rejected_texts = [p + " " + r for p, r in zip(prompts, rejecteds)]
    chosen_enc = tokenizer(chosen_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    rejected_enc = tokenizer(rejected_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    return {
        'chosen_input_ids': chosen_enc['input_ids'],
        'chosen_attention_mask': chosen_enc['attention_mask'],
        'rejected_input_ids': rejected_enc['input_ids'],
        'rejected_attention_mask': rejected_enc['attention_mask'],
        'weights': torch.tensor(weights, dtype=torch.float32)
    }

def m_dpo_loss(r_chosen, r_rejected, weights):
    logits = r_chosen - r_rejected
    loss = -F.logsigmoid(logits)
    return (weights * loss).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--data_path', type=str, default='data/example_data.json')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=2.0)
    args = parser.parse_args()

    with open(args.data_path) as f:
        raw_data = json.load(f)

    cot_sampler = CoTSampler(args.model_name)
    ppl_scorer = PerplexityScorer(args.model_name)
    model = RewardModel(args.model_name).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    augmented_data = []
    for item in raw_data:
        prompt, chosen, rejected = item['prompt'], item['chosen'], item['rejected']
        cot_responses = cot_sampler.sample(prompt, chosen, rejected, num_samples=2)
        for cot in cot_responses:
            g_plus = -ppl_scorer.score(chosen)
            g_minus = -ppl_scorer.score(rejected)
            delta = abs(g_plus - g_minus)
            weight = torch.log(1 + torch.exp(torch.tensor(args.alpha * delta))).item()
            augmented_data.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
                'weight': weight
            })

    dataset = PreferenceDataset(augmented_data, model.tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, model.tokenizer)
    )

    model.train()
    for epoch in range(args.epochs):
        for batch in dataloader:
            chosen_input_ids = batch['chosen_input_ids'].cuda()
            chosen_attn = batch['chosen_attention_mask'].cuda()
            rejected_input_ids = batch['rejected_input_ids'].cuda()
            rejected_attn = batch['rejected_attention_mask'].cuda()
            weights = batch['weights'].cuda()

            r_chosen = model(chosen_input_ids, chosen_attn)
            r_rejected = model(rejected_input_ids, rejected_attn)

            loss = m_dpo_loss(r_chosen, r_rejected, weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'reward_model.bin')

if __name__ == '__main__':
    main()