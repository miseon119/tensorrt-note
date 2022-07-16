# Plugin Basic

## Example

1. Load numpy data

```bash
def createData():
    dataDict = {}
    dataDict[dataName] = np.ones([nDataElement], dtype=np.float32).reshape(4, 4, 4, 4)
    np.savez(dataFile, **dataDict)
    print("Succeeded Saving .npz data!")
    return
```

2. 


