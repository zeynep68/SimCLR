import torchvision.transforms as T


class DataAugmentation:
    """ A stochastic data augmentation module. Returns 2 views of an image. """

    def __init__(self, img_size=96):
        gaussian_blur = T.GaussianBlur(kernel_size=3)

        self.transform = T.Compose([T.ToTensor(), T.RandomResizedCrop(img_size),
                                    T.RandomHorizontalFlip(0.5),
                                    T.RandomApply([gaussian_blur], 0.5),
                                    T.RandomGrayscale(0.2),
                                    T.Normalize(mean=(0.485, 0.456, 0.406),
                                                std=(0.229, 0.224, 0.225))])

    def __call__(self, x):
        return self.transform(x), self.transform(x)
