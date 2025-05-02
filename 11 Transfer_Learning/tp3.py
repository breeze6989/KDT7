import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.ops import box_iou
from PIL import Image
from lxml import etree
from pathlib import Path
from tqdm import tqdm
from effdet import create_model, DetBenchTrain, DetBenchPredict

# -----------------------------------------------------------------------------
# 1) Dataset definition
# -----------------------------------------------------------------------------
class XRAYDataset(Dataset):
    # Define classes and a mapping to class indices
    CLASSES = ['Firecracker', 'Hammer', 'NailClippers', 'Spanner', 'Thinner', 'ZippoOil']
    CLASS2IDX = {c: i+1 for i, c in enumerate(CLASSES)}
    CLASS2IDX['unknown'] = 0  # class 0 for unknown or background
    
    def __init__(self, img_root, xml_root, transform=None):
        self.img_root = Path(img_root)
        self.xml_root = Path(xml_root)
        self.transform = transform
        if not (self.img_root.is_dir() and self.xml_root.is_dir()):
            raise FileNotFoundError(f"Bad paths: {img_root} or {xml_root}")
        # Gather all (image_path, annotation_path) pairs
        self.items = []
        for img_path in sorted(self.img_root.iterdir()):
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                xml_path = self.xml_root / f"{img_path.stem}.xml"
                if not xml_path.exists():
                    raise FileNotFoundError(f"Missing XML for image {img_path.name}")
                self.items.append((img_path, xml_path))
        if not self.items:
            raise ValueError(f"No images found in {img_root}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        img_path, xml_path = self.items[idx]
        # Load image
        image = Image.open(img_path).convert('RGB')
        # Parse XML for annotations
        root = etree.parse(str(xml_path)).getroot()
        boxes = []
        labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text or 'unknown'
            if name not in self.CLASSES:
                name = 'unknown'
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text) // 3
            ymin = int(bbox.find('ymin').text) // 3
            xmax = int(bbox.find('xmax').text) // 3
            ymax = int(bbox.find('ymax').text) // 3
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.CLASS2IDX[name])
        # Convert to tensors
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        sample = {'boxes': boxes_tensor, 'labels': labels_tensor}
        # Apply transforms to image if provided
        if self.transform:
            image = self.transform(image)
        return image, sample

# -----------------------------------------------------------------------------
# 2) Collate function for DataLoader
# -----------------------------------------------------------------------------
def collate_fn(batch):
    # batch is a list of (image, sample) tuples
    images, targets = zip(*batch)
    # Stack images into a single tensor (batch_size, C, H, W)
    images = torch.stack(images, dim=0)
    # Targets remain a list of dictionaries
    return images, list(targets)

# -----------------------------------------------------------------------------
# 3) Validation accuracy calculation
# -----------------------------------------------------------------------------
def val_accuracy(pred, targ, iou_threshold=0.3):
    """
    Compute validation accuracy for one image:
    Returns 1.0 if at least one predicted box matches a ground truth box 
    of the same class with IoU >= threshold, else 0.0 (for images with GT).
    If no ground truth, returns 1.0 (assume correct if nothing to detect);
    if ground truth exists but no prediction, returns 0.0.
    """
    gt_boxes, gt_labels = targ['bbox'], targ['cls']
    # If no ground truth boxes
    if gt_boxes.numel() == 0:
        return 1.0
    pred_boxes, pred_labels = pred['bbox'], pred['cls']
    # If ground truth exists but no predicted boxes
    if pred_boxes.numel() == 0:
        return 0.0
    # Compute IoUs between all predicted and all GT boxes
    ious = box_iou(pred_boxes, gt_boxes)
    # For each GT box, find the prediction with highest IoU
    best_iou, best_idx = ious.max(dim=0)
    # Check if those best predictions match in class and exceed IoU threshold
    matches = (best_iou >= iou_threshold) & (pred_labels[best_idx] == gt_labels)
    # "Accuracy" is the fraction of GT boxes that were matched correctly
    return matches.float().mean().item()

# -----------------------------------------------------------------------------
# 4) Main training and validation loop
# -----------------------------------------------------------------------------
def main():
    # Paths to training and validation image and XML folders (update as needed)
    train_img_dir = r'F:\KDT7\12_trans\team_project\xray-img\Astrophysics\Single_Default' 
    train_xml_dir = r'F:\KDT7\12_trans\team_project\xray-img\Annotation\Train\Pascal\Astrophysics_SingleDefaultOnly2'
    val_img_dir   = r'F:\KDT7\12_trans\team_project\xray-img\Astrophysics_val\Single_Default'
    val_xml_dir   = r'F:\KDT7\12_trans\team_project\xray-img\Annotation\eval\Pascal\Astrophysics_SingleDefaultOnly'
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = (360, 640)  # Expected model input size (H, W)
    
    # Image transformations
    transform = T.Compose([
        T.Resize(input_size),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)  # convert to float tensor (0.0–1.0)
    ])
    
    # Dataset and DataLoader setup
    train_ds = XRAYDataset(train_img_dir, train_xml_dir, transform=transform)
    val_ds   = XRAYDataset(val_img_dir,   val_xml_dir,   transform=transform)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    
    # Create EfficientDet model (Effdet) and bench wrappers
    num_classes = len(XRAYDataset.CLASSES) + 1  # classes + background/unknown
    # Create base model (bench_task=None returns the raw EfficientDet model)
    efficientdet_model = create_model('tf_efficientdet_d0', bench_task=None,
                                      num_classes=num_classes, pretrained=True,
                                      image_size=input_size)
    # Wrap for training and inference
    bench_train = DetBenchTrain(efficientdet_model).to(device)
    bench_predict = DetBenchPredict(efficientdet_model).to(device)
    # (bench_train and bench_predict share the same underlying weights)
    
    optimizer = torch.optim.AdamW(bench_train.parameters(), lr=1e-4)
    best_val_acc = 0.0
    epochs = 50
    for epoch in range(1, epochs + 1):
        # ----- Training -----
        bench_train.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader):
            images = images.to(device)
            # Prepare targets as dict of lists of tensors for DetBenchTrain
            train_targets = {
                'bbox': [t['boxes'].to(device) for t in targets],
                'cls':  [t['labels'].to(device) for t in targets]
            }
            # Forward pass and loss computation
            output = bench_train(images, train_targets)
            loss = output['loss']
            # Backpropagation and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        avg_train_loss = running_loss / len(train_ds)
        
        # ----- Validation -----
        bench_train.eval()  # or bench_predict.eval() since they share weights
        best_val_acc = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader):
                images = images.to(device)
                batch_size = images.size(0)
                # Prepare targets for evaluation (including scale info)
                eval_targets = {
                    'bbox': [t['boxes'].to(device) for t in targets],
                    'cls':  [t['labels'].to(device) for t in targets],
                    'img_scale': torch.ones(batch_size, device=device),
                    'img_size':  torch.tensor([input_size]*batch_size, 
                                               dtype=torch.float32, device=device)
                }
                # Get predictions; use bench_train in eval mode (returns loss & detections)
                output = bench_train(images, eval_targets)
                detections = output['detections'] if isinstance(output, dict) else output
                # Iterate through batch and accumulate accuracy
                for det_tensor, target in zip(detections, targets):
                    # det_tensor shape: (N_pred, 6) -> [x1, y1, x2, y2, score, class]
                    if det_tensor.numel() == 0:
                        # No detections for this image
                        pred_boxes = torch.zeros((0, 4), device=device)
                        pred_labels = torch.zeros((0,), dtype=torch.int64, device=device)
                    else:
                        # Separate boxes and labels from detection tensor
                        pred_boxes = det_tensor[:, :4]
                        pred_labels = det_tensor[:, 5].to(torch.int64)
                    # Ensure 2D shape for comparison (unsqueeze if single box)
                    if pred_boxes.ndim == 1:
                        pred_boxes = pred_boxes.unsqueeze(0)
                    gt_boxes = target['boxes'].to(device)
                    if gt_boxes.ndim == 1:
                        gt_boxes = gt_boxes.unsqueeze(0)
                    # Prepare pred and targ dicts for accuracy calc
                    pred_dict = {'bbox': pred_boxes, 'cls': pred_labels}
                    targ_dict = {'bbox': gt_boxes, 'cls': target['labels'].to(device)}
                    val_correct = val_accuracy(pred_dict, targ_dict)
        
        print(f"Epoch {epoch:02d}: TrainLoss = {avg_train_loss:.4f}, ValAcc = {val_correct:.4f}")
        
        # Save best model weights
        if val_correct > best_val_acc:
            best_val_acc = val_correct
            torch.save(bench_train.state_dict(), "best_model.pth")

if __name__ == "__main__":
    main()
