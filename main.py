from multiprocessing import freeze_support


if __name__ == "__main__":
    freeze_support()

    from raman_classifier.gui import main
    main()
