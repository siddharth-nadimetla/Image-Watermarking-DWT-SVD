Project: DWT-SVD Digital Watermarking & Authentication System

Course: Digital Watermarking and Steganography

Team Members:

Siddharth Nadimetla (Reg No: 22BCI0303)

Kousik Rallapalli (Reg No: 22BCI0308)

Aditya Goyal (Reg No: 22BCI0128)

Rahul Seju (Reg No: 22BCI0314)

1. Project Repository

GitHub Link: https://github.com/siddharth-nadimetla/Image-Watermarking-DWT-SVD

2. Project Description

This project is a Python implementation of a non-blind, frequency-domain digital watermarking system for image authentication. It uses a hybrid DWT-SVD (Discrete Wavelet Transform - Singular Value Decomposition) method to embed a 256-bit SHA-256 hash of a secret message into a color image.

The system is "non-blind," meaning the verification process requires the original cover image and a "key" file (watermark_data.npz) that is generated during embedding.

3. Core Modules (Python Scripts)

The project is divided into two main executable scripts:

main.py (Embedder):

Takes a secret message and a cover image.

Generates a 256-bit hash of the message.

Embeds this hash into the blue channel of the image using DWT-SVD.

Calculates the imperceptibility metrics (PSNR/SSIM).

Output: watermarked_color_image.png and the "key" file watermark_data.npz.

extract.py (Verifier):

Takes a secret message to verify.

Loads the watermarked data and keys from watermark_data.npz.

Loads the original cover_image.png to recalculate the original state.

Extracts the embedded hash and compares it to the expected hash.

Output: A "Success" or "Failure" message.

4. Embedding Methodology

The method is a hybrid frequency-domain technique combining Discrete Wavelet Transform (DWT) and Singular Value Decomposition (SVD).

Step-by-Step Embedding Process

Watermark Preparation:
First, the secret message (e.g., mysecretkey123) is not embedded directly. Instead, it is converted into a secure, fixed-length "fingerprint" using a cryptographic hash function.

Technique: SHA-256

Input: mysecretkey123

Output: A 256-bit binary string (e.g., 10110...), which we'll call W (for Watermark).

Image Preparation:

The color cover image (cover_image.png) is loaded using OpenCV.

It is split into its three color channels: Blue, Green, and Red.

We select the Blue channel for embedding. This is a strategic choice because the Human Visual System (HVS) is less sensitive to changes in the blue channel, which makes the watermark more imperceptible (invisible).

Step 1: Discrete Wavelet Transform (DWT)
We apply a 2-level DWT to the Blue channel. This decomposes the channel's data from the spatial domain (pixels) to the frequency domain (coefficients). This isolates the image's most important, low-frequency components.

Output: The LL2 sub-band. This is a smaller matrix representing the most robust, high-energy, low-frequency approximation of the image. This is where we hide the data.

Step 2: Singular Value Decomposition (SVD)
Next, we apply SVD to this LL2 sub-band. SVD is a matrix factorization technique that breaks a matrix down into its core geometric and structural components.

Formula: LL2 = U * S * V^T

U and V: These are orthogonal matrices that represent the geometric "directions" of the image. We save these as they are the "key" needed for extraction.

S: This is a diagonal matrix containing the singular values. These are the numbers that represent the "energy" or importance of each structural component. These are the values we will modify.

Step 3: Embedding the Watermark (The Core Formula)
We now modify the singular values (S) using the 256 bits from our watermark hash (W). We iterate from i = 0 to 255. The system supports two methods:

Method 1: Additive Embedding (Recommended)

This method adds a fixed, absolute strength (alpha) to the singular value for high robustness.

Formula:
S_modified[i] = S_original[i] + (alpha * W[i])

If the watermark bit W[i] is 0: S_modified[i] = S_original[i] (No change)

If the watermark bit W[i] is 1: S_modified[i] = S_original[i] + alpha

Method 2: Multiplicative Embedding

This method scales the singular value by a percentage, which can be more subtle but less robust.

Formula:
S_modified[i] = S_original[i] * (1 + (alpha * W[i]))

If the watermark bit W[i] is 0: S_modified[i] = S_original[i] (No change)

If the watermark bit W[i] is 1: S_modified[i] = S_original[i] * (1 + alpha)

Reconstruction:
After modifying the first 256 singular values, the system reverses the process:

The modified LL2 sub-band is rebuilt:
LL2_modified = U * S_modified * V^T

The Inverse DWT (IDWT) is applied to LL2_modified to reconstruct the full, watermarked blue channel.

This modified blue channel is merged with the original, untouched green and red channels to create the final watermarked_color_image.png.

5. Setup and Installation

To run this project, you must have Python 3 and the required libraries installed.

Step 1: Create a Virtual Environment (Recommended)

# Create a new virtual environment folder
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate





Step 2: Install Dependencies

Make sure you have a requirements.txt file in your project with the following content:

opencv-python
numpy
PyWavelets
scikit-image





Then, run the following command to install them:

pip install -r requirements.txt





6. Execution of the Code (Step-by-Step Guide)

CRITICAL PREPARATION: The Cover Image

This system is designed to embed a 256-bit hash. Due to the 2-level DWT, the image must be large enough to provide at least 256 singular values.

You must place a large cover image (1024x1024 pixels or larger) in the project folder and name it cover_image.png before you begin.

Part 1: Embedding the Watermark

This step creates the watermarked image and the key file.

Open your terminal or command prompt (with the virtual environment activated).

Run the main.py script:

python main.py





The script will prompt you for the following:

Enter the secret message...: mysecretkey123

Choose an embedding method... (1: Additive): 1

Enter the alpha (e.g., 30): 30

Choose a wavelet... (1: haar): 1

Output: The script will run and generate two new files:

watermarked_color_image.png (The image you can view)

watermark_data.npz (The critical key file)
It will also print the PSNR/SSIM metrics (e.g., PSNR: 38.70 dB).

Part 2: Verifying the Watermark (No Attack)

This step confirms that the watermark was embedded correctly.

Run the extract.py script:

python extract.py





The script will prompt you for the exact same parameters used in Part 1.

Enter the original SECRET message...: mysecretkey123

Choose the embedding method... (1: Additive): 1

Enter the alpha used... (e.g., 30): 30

Output: The script will load the .npz file, perform the verification, and print:

--- Verification Result ---
✅ Success! The watermark is authentic and valid.



