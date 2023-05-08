import coreConfig as cc
from MyModels import *
import MyMetrics as met
import dataset as usd
exec(cc.stmts)

def bestFit(models , inp) :
    interLogits = [] #intermediate logits 
    for model , info in models.items() :
        x = inp[info['spec']].to(device)
        pred = model(x.view(*info['inDim']))
        interLogits.append(pred) 
    resultLogits = torch.stack(interLogits , dim = 0)
    max_values, max_indices = torch.max(resultLogits, dim=0)
    result = torch.squeeze(max_values)
    return result
        

def test_single_epoch(models , data_loader , loss_fun ) :
    for k in models.keys() :
        k.eval() 
    with torch.inference_mode() :
        for inp , tar in data_loader :
            logits = bestFit(models, inp)
            tar = tar.to(torch.int64).to(device)
            loss = loss_fun(logits , tar)
            acc = met.acc(logits , tar)    
            print(f"testing accuracy {acc}")


def getModels():
    models = {}
    for model_name , info in cc.models.items() :
        model = globals()[model_name](*info['params']) if info['params'] else globals()[model_name]()
        if os.path.exists(info['path']):
            status = model.load_state_dict(torch.load(info['path']))
            print(model_name , status)
        model.to(device)
        models.update({model : {"inDim" : info["inDim"] , "spec" : info["spec"]}})
    return models  
    
    

if __name__ == "__main__" :
    print(cc.specSet)
    dataLoader = DataLoader(usd.UrbanSoundDataset(
                spec = cc.specSet,
                train = True,
                test_fold = [1]
                ),
                batch_size=cc.batch_size, shuffle=True , drop_last = cc.drop_last)
    
    
    models = getModels()
    loss_fun = nn.CrossEntropyLoss().to(device)
    test_single_epoch(models , dataLoader , loss_fun )
    
    print("done")
