import os
import secml_malware
from secml.array import CArray
from secml_malware.models.malconv import MalConv
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware, End2EndModel
from secml_malware.attack.whitebox.c_padding_evasion import CPaddingEvasion
from secml_malware.attack.whitebox.c_header_evasion import CHeaderEvasion

net = MalConv()
net = CClassifierEnd2EndMalware(net)
net.load_pretrained_model()

padding_attack = CPaddingEvasion(net, 1000, iterations=1)
header_attack = CHeaderEvasion(net, random_init=False, iterations=20, optimize_all_dos=False, threshold=0.5)
folder = "../sample/malwares"
X = []
y = []
file_names = []
cnt = 0
for i, f in enumerate(os.listdir(folder)):
    path = os.path.join(folder, f)
    
    with open(path, "rb") as file_handle:
        code = file_handle.read()
    x = End2EndModel.bytes_to_numpy(
        code, net.get_input_max_length(), 256, False
    )
    _, confidence = net.predict(CArray(x), True)

    if confidence[0, 1].item() < 0.5:
        continue
    

    print(f"> Added {f} with confidence {confidence[0,1].item()}")
    X.append(x)

    conf = confidence[1][0].item()
    y.append([1 - conf, conf])
    file_names.append(path)
    cnt+=1
    if cnt >= 200:
        break

success=0
for sample, label in zip(X, y):
    y_pred, adv_score, adv_ds, f_obj = header_attack.run(CArray(sample), CArray(label[1]))
    if adv_score[0] < 0.5:
        success+=1
print(f"{success}, rate:{success/200}")