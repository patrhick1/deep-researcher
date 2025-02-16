// static/script.js

// Function to save a file using the File System Access API (supported in Chromium-based browsers)
async function saveFile(content, filename) {
    try {
        const options = {
            suggestedName: filename,
            types: [{
                description: 'Markdown File',
                accept: {'text/markdown': ['.md']},
            }],
        };
        const handle = await window.showSaveFilePicker(options);
        const writable = await handle.createWritable();
        await writable.write(content);
        await writable.close();
    } catch (err) {
        console.error('Error saving file:', err);
    }
}

console.log('Custom script loaded.');
