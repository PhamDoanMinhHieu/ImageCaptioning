import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # Thư viện hỗ trợ NLP 
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Thư viện hỗ trợ xử lí ảnh
import torchvision.transforms as transforms


# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to a index
# 2. We need to setup a Pytorch dataset to load the data
# 3. Setup padding of every batch (all examples should be
#    of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!

# Tải mô hình ngôn ngữ Anh
spacy_eng = spacy.load("en_core_web_sm")

# Định nghĩa lớp Vocabulary để xây dựng bộ từ điển
class Vocabulary:
    def __init__(self,
                 freq_threshold: float # ngưỡng tần số để xác định các từ xuất hiện thường xuyên trong từ điển.
                 ):
        # Các từ đặc biệt trong sử lí ngôn ngữ tự nhiên
        # <PAD> đánh dấu các vị trí không sử dụng, thường được thêm vào để đảm bảo độ dài
        # <SOS> đánh dấu điểm bắt đầu trong câu
        # <EOS> đánh dấu điểm kết thúc trong câu
        # <UNK> đại diện các từ không xuất hiện trong từ điển
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    # Phương thức trả về số lượng từ trong từ điển
    def __len__(self):
        return len(self.itos)

    @staticmethod
    # Phương thức phân tách một câu thành các từ và chuyển thành chữ thường
    def tokenizer_eng(text: str):
        # mô hình SpaCy spacy_eng đã được tải để phân tách một đoạn văn bản tiếng Anh thành các từ riêng lẻ và chuyển chúng về chữ thường.
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    # Phương thức xây dựng từ điển dựa trên danh sách các câu
    def build_vocabulary(self, sentence_list: list):
        
        # tạo một từ điển tần số (frequencies) để đếm tần số xuất hiện của từng từ trong các câu.
        frequencies = {}
        idx = 4

        # Duyệt qua từng câu trong danh sách các câu
        for sentence in sentence_list:
            
            # Phân tách câu thành các từ
            for word in self.tokenizer_eng(sentence):
                
                # Nếu từ không có trong tần số
                if word not in frequencies:
                    
                    # Đặt tần số bằng 1
                    frequencies[word] = 1

                else:
                    # Nếu không thì tăng tần số lên 1
                    frequencies[word] += 1

                # Nếu tần số xuất hiện bằng với ngưỡng
                if frequencies[word] == self.freq_threshold:
                    
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    
    # Phương thức chuyển đổi câu thành danh sách các chỉ số tương ứng
    def numericalize(self, text: str):
        # Đầu tiên, đoạn văn bản được phân tách thành các từ riêng lẻ
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

# Lớp FlickrDataset là một lớp con của lớp Dataset trong PyTorch và
# được sử dụng để tạo một tập dữ liệu từ tệp dữ liệu chứa thông tin về hình ảnh và chú thích tương ứng.
class FlickrDataset(Dataset):
    def __init__(self,
                 root_dir: str, # đường dẫn đến thư mục gốc chứa hình ảnh
                 captions_file: str, # đường dẫn đến tệp chứa thông tin chú thích
                 transform=None, # biến đổi hình ảnh
                 freq_threshold=5 # ngưỡng tần số để xây dựng từ điển
                 ):
               
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Lấy dữ liệu hình ảnh và chú thích
        self.imgs = self.df["image"] # Chứa idx của ảnh
        self.captions = self.df["caption"] # Chứa caption của anh

        # Khởi tạo từ điển và xây dựng từ điển
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    # Phương thức lấy chiều dài dữ liệu
    def __len__(self):
        return len(self.df)

    # Phương thức lấy 1 mẫu dữ liệu: hình ảnh + chú thích
    def __getitem__(self, index: int):
        # Lấy chú thích ảnh
        caption = self.captions[index]
        
        # Lấy đường dẫn của ảnh
        img_id = self.imgs[index]
        
        # Lấy ảnh thông qua đường dẫn
        img_path = str(self.root_dir)  + str(img_id)
        img = Image.open(img_path).convert("RGB")

        # Thực hiện biến đổi ảnh nếu có
        if self.transform is not None:
            img = self.transform(img)

        # (<SOS> <DATA> <EOS>)
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)

# ớp MyCollate được định nghĩa để tùy chỉnh quá trình gom nhóm (collate) các mẫu trong một batch khi sử dụng DataLoader
class MyCollate:
    def __init__(self,
                 pad_idx: int# Để xác định chỉ số sẽ được sử dụng để đánh dấu các vị trí được đệm (padding) trong chuỗi dữ liệu.
                 ):
        self.pad_idx = pad_idx

    def __call__(self,
                 batch # Danh sách các mẫu
                 ):
        
        # Danh sách các hình ảnh trong batch được lấy ra từ mỗi mẫu và mở rộng chiều 0
        # Kết quả là một danh sách các tensor hình ảnh có kích thước (batch_size, C, H, W),
        # trong đó batch_size là số lượng mẫu trong batch, C, H, W lần lượt là số kênh, chiều cao và chiều rộng của hình ảnh.
        imgs = [item[0].unsqueeze(0) for item in batch]
        
        # Tiếp theo, danh sách các tensor hình ảnh được ghép nối theo chiều 0 bằng cách sử dụng torch.cat.
        # Kết quả là một tensor có kích thước (batch_size, C, H, W), đại diện cho tất cả các hình ảnh trong batch.
        imgs = torch.cat(imgs, dim=0)
        
        # Sau đó, danh sách các mục tiêu (targets) trong batch được lấy ra từ mỗi mẫu.
        targets = [item[1] for item in batch]
        
        # Danh sách các mục tiêu được đệm (padding) bằng cách sử dụng hàm pad_sequence,
        # với các tham số targets (danh sách các mục tiêu), batch_first=False (mục tiêu sẽ được đệm theo chiều thứ 2)
        # và padding_value=self.pad_idx (giá trị đệm sẽ là pad_idx).
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        # Cuối cùng, phương thức trả về tensor hình ảnh và tensor các mục tiêu đã được đệm.
        return imgs, targets


def get_loader(
    root_folder: str, # Đường dẫn đến thư mục gốc chứa hình ảnh
    annotation_file: str, # Đường dẫn đến tệp chứa thông tin chú thích
    transform, # Biến đổi hình ảnh
    batch_size=32, # Kích thước batch
    num_workers=4, # Số lượng worker để tải dữ liệu
    shuffle=True, # Xáo trộn dữ liệu
    pin_memory=True,
):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    print("Initialize Dataloder")
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    images_folder = "./data/flickr8k/images/"
    captions_file = "./data/flickr8k/captions.csv"
       
    loader, dataset = get_loader(
        root_folder=images_folder,
        annotation_file=captions_file,
        transform=transform
    )

    print("Loader successfully")
    
    # for idx, (imgs, captions) in enumerate(loader):
    #     print(imgs.shape)
    #     print(captions.shape)
    
    item = dataset.__getitem__(100)
    img, capt = item
    print(img.shape) # (3, 224, 224)
    print(capt.shape)# (15)