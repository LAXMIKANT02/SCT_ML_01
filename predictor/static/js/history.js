function deleteEntry(id) {
    fetch(`/predictor/delete/${id}/`, {
        method: 'POST',
        headers: {
            'X-CSRFToken': getCSRFToken()
        }
    })
    .then(response => {
        if (response.ok) {
            document.getElementById(`row-${id}`).remove();
        } else {
            console.error("Failed to delete entry");
        }
    })
    .catch(error => {
        console.error("Error:", error);
    });
}
function getCSRFToken() {
    const cookie = document.cookie
        .split(';')
        .map(c => c.trim())
        .find(c => c.startsWith('csrftoken='));
    return cookie ? cookie.split('=')[1] : '';
}
