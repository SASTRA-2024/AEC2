import coreConfig as cc
import dataset as usd
import MyMetrics as met
from MyModels import * 
exec(cc.stmts)


inpDim = cc.models[cc.currModel]["inDim"]
def test_single_epoch(model , data_loader , loss_fun , device ) :
    model.eval()
    with torch.inference_mode() :
        for inp , tar in data_loader :
            inp , tar = inp.to(device) , tar.to(device)
            logits = model(inp.view(*inpDim)) 
            
            loss = loss_fun(logits , tar)
            acc = met.acc(logits , tar)
            
        print(f"testing accuracy {acc}")

def train_single_epoch(model , data_loader , loss_fun , optimiser , device ) :
    model.train()
    for inp , tar in data_loader :
        inp , tar = inp.to(device) , tar.to(device)

        logits = model(inp.view(*inpDim))
        print(logits.shape)
        print(logits)
        loss = loss_fun(logits , tar)
        acc = met.acc(logits , tar)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"\ntraining accuracy {acc}")

def fit(model , trLdr , tstLdr  , loss_fun , optimiser , device , epochs):

    for _ in tqdm(range(epochs) , desc = "->"):
        train_single_epoch(model , trLdr , loss_fun , optimiser , device )
        test_single_epoch(model , tstLdr , loss_fun , device)
    

if __name__ == "__main__":
    #idea to perform kfold 
    testFoldSet = [[j+1 for j in i] for i in it.permutations([i for i in range(cc.kfold)], r=cc.num_test_folds)]

    Spec = cc.models[cc.currModel]["spec"]
    ToDB = cc.models[cc.currModel]["toDB"]
    params = cc.models[cc.currModel]["params"]
    pt_file = cc.models[cc.currModel]["path"]
    print(pt_file)
    
    print(testFoldSet)
    loaders = [(DataLoader(usd.UrbanSoundDataset(
                    spec = Spec,
                    toDB = ToDB ,
                    train = True ,
                    test_fold = fold
                    ),
                batch_size=cc.batch_size, shuffle=True , drop_last = cc.drop_last),
                DataLoader(usd.UrbanSoundDataset(
                    spec = Spec,
                    toDB = ToDB ,
                    train = False ,
                    test_fold = fold
                    ),
                batch_size=cc.batch_size, shuffle=True , drop_last = cc.drop_last ))
               for fold in testFoldSet]
    
    model = globals()[cc.currModel](*params) if params else globals()[cc.currModel]()

    if os.path.exists(pt_file):
        status = model.load_state_dict(torch.load(pt_file))
        print(status)
        
    model = model.to(device)
    loss_fun = eval(cc.models[cc.currModel]["loss_fun"])
    optimizer = eval(cc.models[cc.currModel]["optimizer"])

    print(loss_fun , optimizer ,end = "\n\n") 
    

    print("PERFORMING K-FOLDS")
    for i , t in enumerate(loaders) :
        print(f"FOLD {i+1}")
        trLdr , tstLdr = t 
        fit(model , trLdr , tstLdr  , loss_fun , optimizer , device , cc.epochs)
        status = torch.save(model.state_dict(), pt_file)
        print(f"Checkpoint status :{('saved successfully'if status is None else status)}")
    print("training finished")

    
