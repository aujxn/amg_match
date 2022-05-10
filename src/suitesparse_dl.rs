use flate2::read::GzDecoder;
use soup::prelude::*;
use std::io::Read;

pub async fn download_all() -> Result<(), Box<dyn std::error::Error>> {
    let builder = reqwest::ClientBuilder::new();
    let client = builder.http2_max_frame_size(u32::MAX).build()?;
    info!("Getting list of matrices from suitesparse");
    let res = client
        .get("https://sparse.tamu.edu/?per_page=All")
        .header("query", "filterrific[rb_type]: Real")
        .header("query", "filterrific[structure]: Symmetric")
        .header("query", "filterrific[positive_definite]: Yes")
        .send()
        .await?;

    let soup = Soup::new(&res.text().await?);
    let urls: Vec<String> = soup
        .tag("a")
        .find_all()
        .filter_map(|link| link.get("href"))
        .filter(|url| url.contains("https://suitesparse-collection-website.herokuapp.com/MM/HB/"))
        .collect();
    info!("Downloading {} matrices...", urls.len());

    for url in urls.iter() {
        info!("Downloading: {url}");
        let stream = client.get(url).send().await?.bytes().await?;
        let mut gz = GzDecoder::new(&*stream);
        let mut out = Vec::new();
        gz.read_to_end(&mut out)?;
        let mut archive = tar::Archive::new(&out[..]);
        archive.unpack("test_matrices")?;
    }
    Ok(())
}
