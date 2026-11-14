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
  } catch (error) {
    if (project.isCurrent('furniture', token)) {
      project.set('$studioError', error.response?.data?.detail || error.message || '3D generation failed. Please try another image.');
    }
  } finally {
    project.finishRequest('furniture', token);
  }
}
