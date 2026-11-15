import axios from 'axios';
import { blobToDataURL } from './projectStorage.js';

// Owned by the project, rather than the lifetime of a rendered page.
export async function generateFurniture(project, settings, { post = axios.post, fetchImage = fetch } = {}) {
  const token = project.startRequest('furniture');
  project.set('$studioError', null);
  project.set('$saveSuccessMsg', null);
  try {
    let input = settings.image;
    if (!input.startsWith('data:')) {
      const response = await fetchImage(input);
      if (!response.ok) throw new Error('Could not load the reference image');
      input = await blobToDataURL(await response.blob());
    }
    if (!project.isCurrent('furniture', token)) return;
    const response = await post('/furniture/generate-3d', {
      image_base64: input, name: settings.name, category: settings.category,
      remove_bg: settings.removeBg, target_height_m: settings.height,
    });
    if (!project.isCurrent('furniture', token)) return;
    if (response.data?.status !== 'success') throw new Error('3D generation did not return a model.');
    project.set('studio.generatedModel', response.data);
    project.set('studio.libraryType', '');
    project.set('studio.cameraPose', null);
    // Every generated model goes straight into the room library, ready to place in the house.
    const model = response.data;
    try {
      const saved = await post('/furniture/save-to-library', {
        id: model.id, name: model.name, category: model.category, dimensions_m: model.dimensions_m,
        bounds: model.bounds, glb_base64: model.model_glb_base64, preview_base64: model.preview_image_base64,
      });
      if (saved.data?.status === 'success' && saved.data.item) {
        project.set('studio.savedLibraryId', model.id);
        project.set('library.revision', revision => (revision || 0) + 1, 0);
        project.set('$saveSuccessMsg', 'Saved automatically to the room library.');
      }
    } catch (saveError) {
      const detail = saveError.response?.data?.detail;
      project.set('$studioError', `The model was generated but could not be saved to the room library: ${typeof detail === 'string' ? detail : saveError.message}`);
    }
  } catch (error) {
    if (project.isCurrent('furniture', token)) {
      project.set('$studioError', error.response?.data?.detail || error.message || '3D generation failed. Please try another image.');
    }
  } finally {
    project.finishRequest('furniture', token);
  }
}
