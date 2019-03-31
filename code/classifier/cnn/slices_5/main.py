import import_images_and_features

images, features = import_images_and_features.read_images("../../../../data/images/solid-nodules-with-attributes/benigno",
                                        "../../../../data/features/solidNodules.csv")
print(features)
