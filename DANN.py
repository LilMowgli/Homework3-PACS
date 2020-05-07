import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
from reverse_layer import ReverseLayerF
__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, dann_num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
	#label classifier
        self.classifier = nn.Sequential( 
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes), #set to actual number of classes = 7
        ) 
	#domain_classifier
	self.dann_classifier = nn.Sequential(
		nn.Dropout(), 
		nn.Linear(256 * 6 * 6, 4096),
		nn.ReLU(inplace=True),
		nn.Dropout(),
		nn.Linear(4096, 4096),
		nn.ReLU(inplace=True),
		nn.Linear(4096, num_classes), #set to actual number of classes = 2
	)
		
		
    def forward(self, classifier, alpha = None, x):
	if classifier == 'label':
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.label_classifier(x)
	
	elif classifier == 'domain':
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		# gradient reversal layer (backward gradients will be reversed)
            	reverse_features = ReverseLayerF.apply(x, alpha)
            	discriminator_output = self.dann_classifier(reverse_features)
            	return discriminator_output
		
		
        return x


def alexnet(pretrained=True, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict = False)

	#copy label classifier with prerained weights in domain_classifier excluding last FC layer
	for i in range(len(list(model.classifier.children())[:-1])):
		model.dann_classifier[i].weight.data = model.classifier[i].weight.data
		model.dann_classifier[i].bias.data = model.classifier[i].bias.data
	
    return model
