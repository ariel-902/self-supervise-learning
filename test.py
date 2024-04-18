from P2_C import *
import sys

testcsv_path = sys.argv[1]
testfolder_path = sys.argv[2]
output_path = sys.argv[3]

# testcsv_path = './hw1_data_4_students/hw1_data/p2_data/office/val.csv'
# testfolder_path = './hw1_data_4_students/hw1_data/p2_data/office/val/'
# output_path = './p2pred_test.csv'
best_model_filename = "./p2_model.ckpt"

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
test_set = ImgDataset(testfolder_path, tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

model_best = models.resnet50(weights=None)
in_features = model_best.fc.in_features
model_best.fc = nn.Linear(in_features, 65)
model_best.load_state_dict(torch.load(best_model_filename),strict=False)
model_best = model_best.to(device)

model_best.eval()
prediction = []
filename = []
with torch.no_grad():
    for data,_ ,fname in tqdm(test_loader):
        # print(fname)
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()
        filename += fname

files = []
for file in filename:
    file = file.split("/")[-1]
    files.append(file)

result = dict(zip(files, prediction))

result_label = []
df = pd.read_csv(testcsv_path,index_col=False)
# print(df)
for index, contents in df.iterrows():
    # print(contents["filename"])
    result_label.append(result[contents["filename"]])
    # print(result[contents["filename"]])
# df["filename"] = files
df["label"] = result_label
# right_num = 0
# for index, contents in df.iterrows():
#    right = int(contents["filename"].split("_")[0])
#    if(right == contents["label"]):
#        right_num += 1
# print("accuracy:")
# print("{:.5f}".format(right_num/len(df.index)))
# "{:.2f}".format(pi)

# with pd.option_context('display.max_rows', None,):
#     print(df)
# df["label"]
df.to_csv(output_path,index = False)



