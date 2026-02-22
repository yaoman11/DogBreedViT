from transformers import ViTForImageClassification


def build_vit_model(num_classes=10, freeze_ratio=0.75):
    """
    Membangun model Vision Transformer (ViT-B/16) pretrained dari ImageNet-21k
    dengan fine-tuning parsial.

    Args:
        num_classes (int): jumlah kelas output.
        freeze_ratio (float): persentase layer encoder yang di-freeze (0.0–1.0).
    """
    # Load model pretrained
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    # Partial Freezing Encoder
    total_layers = len(model.vit.encoder.layer)
    freeze_until = int(total_layers * freeze_ratio)

    print(f"Freezing {freeze_until}/{total_layers} encoder layers ({freeze_ratio*100:.0f}%)...")

    for i, layer in enumerate(model.vit.encoder.layer):
        if i < freeze_until:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

    # Freeze embeddings jika freeze_ratio > 80%
    if freeze_ratio > 0.8:
        for param in model.vit.embeddings.parameters():
            param.requires_grad = False
    else:
        for param in model.vit.embeddings.parameters():
            param.requires_grad = True

    return model
