import file_operation as fo
import neuralnet as NNet

eps = 0.03

def main(data,data_name):
    nn = NNet.NeuralNet(64, 30, 20)
    ev_first = nn.evaluate(data)
    n = ev_first/30
    ev = ev_first
    print("--------------学習中-------------")
    while( eps < ev ):
        nn.learn(data)
        ev = nn.evaluate(data)
        num = int((ev_first-ev)/(n))
        print("\r"+"|"+"■"*num+"—"*(30-num-1)+" |",end="")

    print("学習終了")
    if data_name == "ldata_1":
        print("")
        print("--------------筆者0の学習データにおける学習--------------")
        print("筆者0の学習用データの識別結果(%)：",nn.test(fo.ldata_1))
        print("筆者0のテスト用データの識別結果(%)：",nn.test(fo.tdata_1))
        print("筆者1のテスト用データの識別結果(%)：",nn.test(fo.tdata_2))
    elif data_name == "ldata_2":
        print("")
        print("--------------筆者1の学習データにおける学習--------------")
        print("筆者1の学習用データの識別結果(%)：",nn.test(fo.ldata_2))
        print("筆者0のテスト用データの識別結果(%)：",nn.test(fo.tdata_1))
        print("筆者1のテスト用データの識別結果(%)：",nn.test(fo.tdata_2))
    else:
        print("")    
        print("--------------筆者0と筆者1の学習データにおける学習--------------")
        print("筆者0と筆者1の学習用データの識別結果(%)：",nn.test(fo.ldata))
        print("筆者0と筆者1のテスト用データの識別結果(%)：",nn.test(fo.tdata))
        
    print("「あ」の出力ユニット：",nn.backforward(fo.ldata_1[0][0]))


if __name__== '__main__':

    fo.load()
    main(fo.ldata_1,"ldata_1")
    main(fo.ldata_2,"ldata_2")
    main(fo.ldata,"ldata")
    