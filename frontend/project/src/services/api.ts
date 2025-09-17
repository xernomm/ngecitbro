const API_BASE_URL = 'http://localhost:8000';

// Tipe baru untuk respons upload yang lebih deskriptif, sesuai dengan output backend.
interface UploadResponse {
  status: 'ok' | 'exists' | 'error';
  doc_id: string;
  message?: string;
  saved_to: string;
  index_result?: any;
}

export class ApiService {
  /**
   * Mengunggah file (PDF, DOCX, TXT, dll.) ke backend untuk diproses dan diindeks.
   * @param file File yang akan diunggah.
   * @param docId ID dokumen opsional.
   * @returns Promise yang berisi detail hasil upload.
   */
  static async uploadDocument(file: File, docId?: string): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (docId) {
      formData.append('doc_id', docId);
    }

    const response = await fetch(`${API_BASE_URL}/contexts/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      // Peningkatan: Mencoba membaca pesan error detail dari body respons JSON.
      // Ini sangat berguna untuk menangkap error validasi dari FastAPI (misal: tipe file tidak didukung).
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Upload failed: ${errorData.detail || response.statusText}`);
    }

    return response.json();
  }

  /**
   * Menambahkan konteks teks manual ke backend.
   * Catatan: Pastikan backend Anda memiliki endpoint '/contexts/manual'.
   * @param text String teks yang akan ditambahkan.
   * @returns Promise yang berisi respons dari server.
   */
  static async addManualContext(text: string) {
    const response = await fetch(`${API_BASE_URL}/contexts/manual`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Manual context upload failed: ${errorData.detail || response.statusText}`);
    }

    return response.json();
  }

  /**
   * Mengambil daftar semua dokumen yang tersedia dari backend.
   * @returns Promise yang berisi daftar ID dokumen.
   */
  static async listDocuments() {
    const response = await fetch(`${API_BASE_URL}/contexts`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Failed to list documents: ${errorData.detail || response.statusText}`);
    }

    // parse JSON once
    const data = await response.json();

    console.log(data)
    // server returns { contexts: [ {doc_id, filename}, ... ] }
    return data.contexts ?? [];
  }

  /**
   * Menghapus dokumen berdasarkan ID-nya.
   * @param docId ID dokumen yang akan dihapus.
   * @returns Promise yang berisi konfirmasi dari server.
   */
  static async deleteDocument(docId: string) {
    // Peningkatan: Menggunakan encodeURIComponent untuk memastikan ID yang mengandung karakter khusus aman di URL.
    const response = await fetch(`${API_BASE_URL}/contexts/${encodeURIComponent(docId)}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Failed to delete document: ${errorData.detail || response.statusText}`);
    }

    return response.json();
  }

  /**
   * Mengambil potongan konteks yang relevan dari dokumen tertentu berdasarkan query.
   * @param docId ID dokumen yang akan dicari.
   * @param query Pertanyaan atau teks untuk pencarian relevansi.
   * @param nResults Jumlah hasil yang diinginkan.
   * @returns Promise yang berisi hasil pencarian.
   */
  static async retrieveContext(docId: string, query: string, nResults = 5) {
    const params = new URLSearchParams({
      q: query,
      n_results: nResults.toString(),
    });

    const response = await fetch(`${API_BASE_URL}/contexts/${encodeURIComponent(docId)}/retrieve?${params}`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Failed to retrieve context: ${errorData.detail || response.statusText}`);
    }

    return response.json();
  }

  /**
   * Mengirim pesan chat ke backend untuk mendapatkan respons dari model.
   * @param request Objek permintaan chat.
   * @returns Promise yang berisi respons dari chat model.
   */
  static async sendChatMessage(request: any) {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Chat request failed: ${errorData.detail || response.statusText}`);
    }

    return response.json();
  }

  /**
   * Memeriksa status kesehatan (health check) dari API backend.
   * @returns Promise yang bernilai true jika backend sehat, false jika tidak.
   */
  static async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return response.ok;
    } catch (error) {
      return false;
    }
  }
}
