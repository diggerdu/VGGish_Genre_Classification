from graphs.models import SpatialGroupEnhance, DenseBlock, NestedUNet, MelNet
import torch




mn = MelNet().cuda()
mel_in = torch.ones((3,1,512,512)).cuda()
mel_out = mn(mel_in)
__import__('pdb').set_trace() 
nb = NestedUNet().cuda()
nb_out = nb(torch.ones((1,1,512,512)).cuda())


print(nb_out)
db_test = DenseBlock(10, 64, 12)
print(db_test(torch.ones(12,64,32,32)).shape)
sge =  SpatialGroupEnhance(64)
sge(torch.ones((3,64,64,64))).shape


