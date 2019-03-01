from fastai.vision import *
from fastai.data_block import data_collate
import pdb
import sys

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(42)

def get_training_validation_df(train_df, valid_pct=0.1):
    """
    Create random training and validation df, with 
    non-overlapping images. Whales with only single images
    goes to training, since during evaluation
    """
    # whale ids : image filenames
    w2fnames = collections.defaultdict(list)
    for w, f in zip(train_df.Id.values, train_df.Image.values): w2fnames[w].append(f)
    # Validation fnames
    train_fnames = []
    valid_fnames = []
    for _, fnames in w2fnames.items():
        if len(fnames) == 1:
            train_fnames += fnames
        else:
            n = len(fnames)
            n_val = max(1, int(n*valid_pct)) 
            fnames = np.random.permutation(fnames)
            train_fnames += list(fnames[n_val:])
            valid_fnames += list(fnames[:n_val])
    training_df = train_df[train_df.Image.isin(train_fnames)].reset_index(drop=True)
    validation_df = train_df[train_df.Image.isin(valid_fnames)].reset_index(drop=True)

    print(f"unique classes in train : {training_df['Id'].nunique()}")
    print(f"unique classes in valid: {validation_df['Id'].nunique()}")
    print(f"unique classes with single image: {training_df['Id'].nunique() - validation_df['Id'].nunique()}")
    assert len(training_df) + len(validation_df) == len(train_df)
    return training_df, validation_df


class ImageTuple(ItemBase):
    def __init__(self, img1, img2):
        self.img1,self.img2 = img1,img2
        self.obj,self.data = (self.img1,self.img2),[self.img1.data,self.img2.data]
        self.stats = torch.tensor(imagenet_stats)
        
    def apply_tfms(self, tfms, **kwargs):
        self.img1 = self.img1.apply_tfms(tfms, **kwargs)
        self.img2 = self.img2.apply_tfms(tfms, **kwargs)
        self.data = [self.img1.data, self.img2.data]
        self.data = [normalize(t, *self.stats) for t in self.data]
        return self
    
    def to_one(self): 
        """rather than in reconstruct() denormalizations for show methods happen here"""
        return Image(torch.cat([denormalize(t, *self.stats) for t in self.data],2))
    
    def __repr__(self):
        return f'{self.__class__.__name__}{(self.img1.shape, self.img2.shape)}'
        

class ImageTupleList(ImageList):
    def __init__(self, items, **kwargs):
        """items should be tuple of image paths"""
        super().__init__(items, **kwargs)
    
    def get(self, i):
        try:
            fn1, fn2 = self.items[i]
        except:
            pdb.post_mortem(sys.last_traceback)
        img1, img2 = open_image(fn1), open_image(fn2)
        return ImageTuple(img1, img2)
    
    def reconstruct(self, t:Tensor): 
        return ImageTuple(Image(t[0]),Image(t[1]))
    
    @classmethod
    def from_folders(cls, path, folderA, folderB, **kwargs):
        itemsB = ImageList.from_folder(path/folderB).items
        res = super().from_folder(path/folderA, itemsB=itemsB, **kwargs)
        res.path = path
        return res
    
    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(12,6), **kwargs):
        "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, y=ys[i], **kwargs)
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, figsize:Tuple[int,int]=None, **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
        `kwargs` are passed to the show method."""
        figsize = ifnone(figsize, (12,3*len(xs)))
        fig,axs = plt.subplots(len(xs), 2, figsize=figsize)
        fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
        for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
            x.to_one().show(ax=axs[i,0],y=y, **kwargs)
            x.to_one().show(ax=axs[i,1],y=z, **kwargs)

    def analyze_pred(self, pred, thresh:float=0.5): 
        return (pred >= thresh).float()

def grab_image_files(data_dir): 
    return [o for o in Path(data_dir).glob("**/*") if o.suffix in ['.png','.jpg']]

def create_matching_df(df, label_col="Id" , image_col="Image", exclude=None, sample=5000):
    """
    Create tuple of matching image pairs based on their labels
    """
    if exclude: df = df[~(df[label_col].isin(exclude))].reset_index(drop=True).copy()
    c2fnames = dict(df.groupby(label_col).apply(lambda x: list(x[image_col])))
    c2fnames = {c:fnames for c, fnames in c2fnames.items() if len(fnames)>1}
    matching_A = []
    matching_B = []
    labels = []
    for i,c in enumerate(c2fnames):
        fnames = np.array(c2fnames[c])
        shuffled_fnames = np.random.permutation(fnames)
        while np.any(fnames == shuffled_fnames):
            shuffled_fnames = np.random.permutation(shuffled_fnames)
        matching_A += list(fnames)
        matching_B += list(shuffled_fnames)
        labels += [c]*len(fnames)
    # check if filenames are not overlapping
    assert np.all(np.array(matching_A) != np.array(matching_B))
    matching_df = pd.DataFrame({"ImageA": matching_A, "ImageB": matching_B, label_col:labels})
    if sample:
        matching_df = matching_df.sample(n=sample).reset_index(drop=True)
    return matching_df

def create_random_non_matching_df(df, label_col="Id", image_col="Image", exclude=None, sample=5000):
    """
    Randomly permute images and make sure there is no class overlap
    """
    if exclude: df = df[~(df[label_col].isin(exclude))].reset_index(drop=True).copy()
    dfA = df.copy().rename(columns={image_col:"ImageA", label_col:"IdA"})
    dfB = (df.iloc[np.random.permutation(df.index)].reset_index(drop=True)
           .rename(columns={image_col:"ImageB", label_col:"IdB"}))
    dfAB = pd.concat([dfA, dfB], 1)
    dfAB_non_matching = dfAB[(dfAB["ImageA"].values != dfAB["ImageB"].values) 
                                                & (dfAB["IdA"].values != dfAB["IdB"].values)]
    non_matching_df = pd.DataFrame({"ImageA":list(dfAB_non_matching["ImageA"]), 
                                                 "ImageB":list(dfAB_non_matching["ImageB"])})
    if sample:
        non_matching_df = non_matching_df.sample(n=sample).reset_index(drop=True)
    return non_matching_df

def concat_matching_nonmatching(matching_df, non_matching_df):
    """
    create concatenated dataframe of matching an non-matchin pairs
    also return targets {1: match, 0: no match}
    """
    imageA = np.concatenate([matching_df['ImageA'], non_matching_df['ImageA']])
    imageB = np.concatenate([matching_df['ImageB'], non_matching_df['ImageB']])
    targets = [1]*len(matching_df) + [0]*len(non_matching_df)
    pair_df = pd.DataFrame({"ImageA":imageA, "ImageB":imageB, "IsMatch":targets})
    return pair_df 

def create_pair_databunch(itemsA, itemsB, targets, bs=64, size=(112,112), valid_pct=0.2):
    """
    ImageA: item A
    ImageB: item B
    """
    itemlist = ImageTupleList(list(zip(itemsA, itemsB)))
    n = len(targets)
    
    # random split
    shuffled_idxs = np.random.permutation(range(n))
    n_val = max(1, int((n*valid_pct)))
    val_idxs, trn_idxs = shuffled_idxs[:n_val], shuffled_idxs[n_val:]
    itemlists = itemlist.split_by_idxs(trn_idxs, val_idxs)
    labellists = itemlists.label_from_lists(targets[trn_idxs], targets[val_idxs])
    data = (labellists.transform(get_transforms(max_rotate=10, p_affine=0.5, p_lighting=0.5,
                                                max_warp=0, max_zoom=0),
                                                size=size,
                                                resize_method=ResizeMethod.SQUISH)
                                                .databunch(bs=bs))
    return data 

def create_unshuffled_dl(df, path, folder, bs=64, size=(112, 112)):
    """
    Create unshuffled and un-transformed dataloaders for vocab and query embedding precalculation
    """
    data = (ImageList.from_df(df, path=path, folder=folder)
            .no_split()
            .label_const()
            .transform(None, size=size, resize_method=ResizeMethod.SQUISH))
 
    dl = DeviceDataLoader(DataLoader(data.train, shuffle=False, batch_size=bs), device=0)
    return dl

class PairEmbeddingDataset(Dataset):
    """custom dataset to create batch of all embedding pairs"""
    def __init__(self, embeddings, pair_idxs):
        self.embeddings = embeddings
        self.pair_idxs = pair_idxs
    
    def __getitem__(self, index):
        idx1, idx2 = self.pair_idxs[index]
        return [self.embeddings[idx1], self.embeddings[idx2]], 0.
        
    def __len__(self):
        return len(self.pair_idxs)
