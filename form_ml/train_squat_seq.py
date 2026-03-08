import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from form_ml.dataset import SquatSeqDataset, collate_pad
from form_ml.model import LSTMClassifier

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SquatSeqDataset(
        features_dir="outputs/squat_day1",
        labels_csv="data/squat_labels.csv",
    )

    # 80/20 split
    n_total = len(dataset)
    n_test = max(1, int(0.2 * n_total))
    n_train = n_total - n_test
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_pad)
    test_loader  = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_pad)

    # Infer input_dim from one sample
    x0, y0, _ = dataset[0]
    input_dim = x0.shape[1]

    model = LSTMClassifier(input_dim=input_dim, hidden_dim=128, num_layers=1, num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train
    model.train()
    for epoch in range(15):
        total_loss = 0.0
        for x, lengths, y, _ in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward() 
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d} | loss={total_loss/len(train_loader):.4f}")

    # Eval
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, lengths, y, _ in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            logits = model(x, lengths)
            pred = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.tolist())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n=== Sequence Model Evaluation (Held-out Test) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nReport:\n", classification_report(y_true, y_pred, digits=4))

    torch.save(model.state_dict(), "outputs/squat_day1/squat_seq_lstm.pt")
    print("\nSaved model to outputs/squat_day1/squat_seq_lstm.pt")

if __name__ == "__main__":
    main()