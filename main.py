import torch
import train
import mnist
import mnistm
import model


def main():
    source_train_loader = mnist.mnist_train_loader
    target_train_loader = mnistm.mnistm_train_loader

    if torch.cuda.is_available():
        encoder = model.Extractor3D().cuda()
        classifier = model.Classifier3D().cuda()
        discriminator = model.Discriminator3D().cuda()

        train.source_only(encoder, classifier, source_train_loader, target_train_loader)
        train.dann(encoder, classifier, discriminator, source_train_loader, target_train_loader)
    else:
        print("No GPUs available.")


if __name__ == "__main__":
    main()
