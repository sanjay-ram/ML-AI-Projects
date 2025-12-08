from nlpaug.augmenter.word import BackTranslationAug
import torch

def main():
    # Initialize the backtranslation augmenter (English -> French -> English)
    back_translation_aug = BackTranslationAug(
        from_model_name='facebook/wmt19-en-de', 
        to_model_name='facebook/wmt19-de-en',
        device= 'cpu'
    )

    # Example text to augment
    text = "The weather is great today."

    # Perform backtranslation to create augmented text
    augmented_text = back_translation_aug.augment(text)

    print("Original text:", text)
    print("Augmented text:", augmented_text)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
