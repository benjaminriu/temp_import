#!/usr/bin/env python3
import requests, sys
def main(argv):
    if len(argv) == 1:
        target_repository = argv[0]
    else:
        target_repository = "../preprocessed_datasets/"
    url_start = "https://github.com/anonymousNeurIPS2021submission5254/SupplementaryMaterial/raw/main/preprocessed_datasets/"
    file_type = ".npy"
    for file_task in ["regression", "classification"]:
        for i in range(16):
            file_name = file_task + str(i) + file_type
            full_url = url_start + file_name
            r = requests.get(full_url, allow_redirects=True)
            open(target_repository+file_name, 'wb').write(r.content)
if __name__ == "__main__":
    main(sys.argv[1:])