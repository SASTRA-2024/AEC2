#DATASET
import coreConfig as cc
exec(cc.stmts)


class UrbanSoundDataset(Dataset) :    
    mfcc_t = cc.mfccTransform
    lfcc_t = cc.lfccTransform
    mel_t = cc.melTransform
    amp_to_DB = T.AmplitudeToDB().to(device)
    
    def __init__(self, toDB ,spec = None , train = True, test_fold = [1]  , kfold = 10) :
        self.annotation = pd.read_csv(cc.annot_file)
        self.audio_dir = cc.audio_dir
        self.device = device 
        self.target_sample_rate = cc.sample_rate
        self.num_samples = cc.num_samples
        self.train = train
        self.spec = spec.strip() if spec else None 
        self.toDB = toDB

        self.trainSet = [i for i in range(len(self.annotation)) if int(self.annotation.iloc[i ,5]) not in test_fold]
        self.testSet = [i for i in range(len(self.annotation)) if int(self.annotation.iloc[i ,5]) in test_fold]
        
    def __len__(self) :
        return len(self.trainSet) if self.train else len(self.testSet)

    def __getitem__(self,index):

        index = self.trainSet[index] if self.train else self.testSet[index]                                            
    
        audio_sample_path = self.get_audio_sample_path(index)  
        label = self.get_audio_sample_label(index)             
        signal , sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self.resampleIfNecessary(signal, sr)
        signal = self.mixDownIfNecessary(signal)
        signal = self.cutIfNecessary(signal)
        signal = self.rightPadIfNecessary(signal)

        
        mfcc = self.mfcc_t(signal)
        if self.toDB.get("mfcc",False) :
            mfcc = self.amp_to_DB(mfcc)
        
        lfcc = self.lfcc_t(signal)
        if self.toDB.get("lfcc",False) :
            lfcc = self.amp_to_DB(lfcc)
        
        mel = self.mel_t(signal)
        if self.toDB.get("mel",False) :
            mel = self.amp_to_DB(mel)
        
        d = {"mfcc" : mfcc , "lfcc" : lfcc , "mel" : mel}
        return d.get(self.spec,d) , label

    def cutIfNecessary(self , signal) :
        if signal.shape[1] > self.num_samples :
            signal = signal[:,:self.num_samples]
        return signal

    def rightPadIfNecessary(self , signal) :
        length_signal = signal.shape[1]
        if length_signal < self.num_samples :
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0 , num_missing_samples)
            signal = F.pad(signal , last_dim_padding)
        return signal 
            
        
    def resampleIfNecessary(self, signal, sr) :
        if sr != self.target_sample_rate:
            resampler = T.Resample(sr , self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def mixDownIfNecessary(self,signal):
        return torch.mean(signal , dim = 0 , keepdim = True) if signal.shape[0] > 1 else signal
        


    def get_audio_sample_path(self,index: int) :
        
        fold = f"fold{self.annotation.iloc[index ,5]}"                      
        path = os.path.join(self.audio_dir , fold , self.annotation.iloc[index , 0])
        return path

    def get_audio_sample_label(self, index : int) :
        return self.annotation.iloc[index , 6]



if __name__ == "__main__":
    #idea to perform kfold

    ds = UrbanSoundDataset(
                    spec = "mel" ,
                    toDB = cc.models[cc.currModel]["toDB"],
                    train = True ,
                    test_fold = [1]
                    )

    print(ds[0])
    testFoldSet = [[j+1 for j in i] for i in it.permutations([i for i in range(cc.kfold)], r=cc.num_test_folds)]

    print(testFoldSet)
    loaders = [(DataLoader(UrbanSoundDataset(
                    spec = cc.models[cc.currModel]["spec"],
                    toDB = cc.models[cc.currModel]["toDB"],
                    train = True ,
                    test_fold = t
                    ),
                batch_size=cc.batch_size, shuffle=True),
                DataLoader(UrbanSoundDataset(
                    spec = cc.models[cc.currModel]["spec"],
                    toDB = cc.models[cc.currModel]["toDB"],
                    train = False ,
                    test_fold = t
                    ),
                batch_size=cc.batch_size, shuffle=True)) for t in testFoldSet]    


    print(f"total loaders {len(loaders)*2}")  
    print(device)
