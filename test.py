from glob import glob
import torch
import numpy as np
import imageio.v2 as imageio
from model import Srnet

TEST_BATCH_SIZE = 16
COVER_PATH = "C:\\Users\\adora\\Desktop\\FYP\\test5\\dataset\\test\\cover\\*.pgm"  # or any other image extension
STEGO_PATH = "C:\\Users\\adora\\Desktop\\FYP\\test5\\dataset\\test\\stego\\*.pgm"  # or any other image extension
CHKPT = "C:\\Users\\adora\\Desktop\\FYP\\test5\\checkpoints\\net_100.pt"

cover_image_names = glob(COVER_PATH)
stego_image_names = glob(STEGO_PATH)

cover_labels = np.zeros((len(cover_image_names)))
stego_labels = np.ones((len(stego_image_names)))

model = Srnet().cuda()

ckpt = torch.load(CHKPT)
model.load_state_dict(ckpt["model_state_dict"])

images = torch.empty((TEST_BATCH_SIZE, 1, 256, 256), dtype=torch.float)
test_accuracy = []

for idx in range(0, len(cover_image_names), TEST_BATCH_SIZE // 2):
    cover_batch = cover_image_names[idx : idx + TEST_BATCH_SIZE // 2]
    stego_batch = stego_image_names[idx : idx + TEST_BATCH_SIZE // 2]

    batch = []
    batch_labels = []

    xi = 0
    yi = 0
    for i in range(2 * len(cover_batch)):
        if i % 2 == 0:
            batch.append(stego_batch[xi])
            batch_labels.append(1)
            xi += 1
        else:
            batch.append(cover_batch[yi])
            batch_labels.append(0)
            yi += 1

    for i in range(TEST_BATCH_SIZE):
        img = imageio.imread(batch[i])
        if img.ndim == 3 and img.shape[2] == 3:  # Convert RGB to grayscale
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        img = torch.tensor(img, dtype=torch.float).cuda()
        img = torch.nn.functional.interpolate(img.unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        images[i, 0, :, :] = img

    image_tensor = images.cuda()
    batch_labels = torch.tensor(batch_labels, dtype=torch.long).cuda()

    outputs = model(image_tensor)
    prediction = outputs.data.max(1)[1]

    accuracy = (
        prediction.eq(batch_labels.data).sum()
        * 100.0
        / (batch_labels.size()[0])
    )
    test_accuracy.append(accuracy.item())

print(f"test_accuracy = {sum(test_accuracy)/len(test_accuracy):.2f}")


'''# Constants
TEST_BATCH_SIZE = 40
COVER_PATH = "C:\\Users\\adora\\Desktop\\FYP\\test4\\dataset\\test\\cover"
STEGO_PATH = "C:\\Users\\adora\\Desktop\\FYP\\test4\\dataset\\test\\stego"
CHKPT = "./checkpoints/net_50.pt"

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
test_dataset = dataset.DatasetLoad(COVER_PATH, STEGO_PATH, size=40)  # Set the correct size of your test dataset
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

# Load model
model = GNCNN().to(device)
ckpt = torch.load(CHKPT)
model.load_state_dict(ckpt["model_state_dict"])

# List to store accuracies
test_accuracy = []

# Test the model
model.eval()  # Set model to evaluation mode

with torch.no_grad():  # Disable gradient calculation
    for batch in test_loader:
        cover_images = batch['cover'].unsqueeze(1).float().to(device)
        stego_images = batch['stego'].unsqueeze(1).float().to(device)
        labels = torch.cat(batch['label']).to(device).view(-1)

        images = torch.cat((cover_images, stego_images), 0)
        outputs = model(images)
        prediction = outputs.data.max(1)[1]

        accuracy = prediction.eq(labels.data).sum().item() * 100.0 / labels.size(0)
        test_accuracy.append(accuracy)

# Ensure there are accuracy values before calculating the mean
if len(test_accuracy) == 0:
    print("No accuracy values calculated. Check your data and batching logic.")
else:
    # Print results
    print(f"Total Test Accuracy Sum: {sum(test_accuracy)}")
    print(f"Total Test Accuracy Count: {len(test_accuracy)}")
    print(f"Test Accuracy = {sum(test_accuracy)/len(test_accuracy):.2f}%")
'''