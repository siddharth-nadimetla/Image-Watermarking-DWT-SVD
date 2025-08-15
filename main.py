import cv2
import numpy as np
import pywt

def text_to_binary(text):
    """Converts a string of text to a binary string."""
    return ''.join(format(ord(char), '08b') for char in text)

def apply_dwt_svd(image, wavelet='haar'):
    """Applies a 2-level DWT and then SVD to the LL2 sub-band."""
    coeffs = pywt.wavedec2(image, wavelet, level=2)
    LL2, (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs
    U, S, V = np.linalg.svd(LL2)
    return (coeffs, U, S, V)

def embed_watermark(S, watermark_binary, alpha=0.1):
    """Embeds the watermark into the singular values."""
    S_modified = np.copy(S)
    watermark_len = len(watermark_binary)
    if watermark_len > len(S):
        raise ValueError("Watermark is too long for the given image.")
    for i in range(watermark_len):
        bit = int(watermark_binary[i])
        S_modified[i] = S[i] * (1 + alpha * bit)
    return S_modified

def reconstruct_image(coeffs, U, S_modified, V, wavelet='haar'):
    """Reconstructs the image from DWT and SVD components."""
    _, (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs
    LL2_modified = U @ np.diag(S_modified) @ V
    coeffs_modified = [LL2_modified, (LH2, HL2, HH2), (LH1, HL1, HH1)]
    reconstructed_image = pywt.waverec2(coeffs_modified, wavelet)
    reconstructed_image = np.clip(reconstructed_image, 0, 255)
    reconstructed_image = np.uint8(reconstructed_image)
    return reconstructed_image

# --- Main Execution with User Input ---
if __name__ == "__main__":
    # --- 1. Get User Input ---
    print("--- Digital Watermarking Embedding ---")
    
    # Get Watermark Text
    watermark_text = input("Enter the watermark text: ")
    
    # Get Alpha Value with Error Handling
    while True:
        try:
            alpha_str = input("Enter the alpha (embedding strength, e.g., 0.05): ")
            alpha = float(alpha_str)
            if alpha > 0:
                break
            else:
                print("Alpha must be a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number for alpha.")

    # Get Wavelet Choice
    available_wavelets = ['haar', 'db1', 'bior1.3', 'sym2']
    print("\nAvailable wavelets:")
    for i, w in enumerate(available_wavelets):
        print(f"  {i+1}: {w}")
    
    while True:
        try:
            choice_str = input(f"Choose a wavelet (1-{len(available_wavelets)}): ")
            choice = int(choice_str)
            if 1 <= choice <= len(available_wavelets):
                wavelet = available_wavelets[choice - 1]
                break
            else:
                print("Invalid choice. Please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    print(f"\nConfiguration: Text='{watermark_text}', Alpha={alpha}, Wavelet='{wavelet}'")
    
    # --- 2. Process the Image ---
    try:
        cover_image_original = cv2.imread('cover_image.png', cv2.IMREAD_GRAYSCALE)
        if cover_image_original is None:
            raise FileNotFoundError("Image not found. Make sure 'cover_image.png' is in the directory.")
        h, w = cover_image_original.shape
        cover_image = cover_image_original[:h - h % 4, :w - w % 4]
    except FileNotFoundError as e:
        print(e)
        exit()

    watermark_binary = text_to_binary(watermark_text)
    
    # Pass the chosen wavelet to the functions
    coeffs, U, S, V = apply_dwt_svd(cover_image, wavelet=wavelet)
    
    try:
        S_watermarked = embed_watermark(S, watermark_binary, alpha)
    except ValueError as e:
        print(e)
        exit()
        
    watermarked_image = reconstruct_image(coeffs, U, S_watermarked, V, wavelet=wavelet)
    
    # --- 3. Save and Display Results ---
    cv2.imwrite('watermarked_image.png', watermarked_image)
    print("\nSuccessfully embedded watermark. 'watermarked_image.png' saved.")
    
    cv2.imshow('Original Cover Image', cover_image)
    cv2.imshow('Watermarked Image', watermarked_image)
    
    print("Displaying images. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()