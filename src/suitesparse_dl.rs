//! Downloader tool to get all the s.p.d. matrices from TAMU's
//! suitesparse collection

use flate2::read::GzDecoder;
use soup::prelude::*;
use std::io::Read;

pub async fn download_all() -> Result<(), Box<dyn std::error::Error>> {
    let builder = reqwest::ClientBuilder::new();
    let client = builder.http2_max_frame_size(u32::MAX).build()?;
    info!("Getting list of matrices from suitesparse");
    let url = "https://sparse.tamu.edu/?per_page=All".to_string();
    let args = "&filterrific[sorted_by]=id_asc&filterrific[rb_type]=Real&filterrific[positive_definite]=Yes";
    let res = client.get(url + args).send().await?;

    let soup = Soup::new(&res.text().await?);
    let results = soup.attr("id", "matrices").find().unwrap();
    let urls: Vec<String> = results
        .tag("a")
        .find_all()
        .filter_map(|link| link.get("href"))
        .filter(|url| url.contains("https://suitesparse-collection-website.herokuapp.com/MM/"))
        .collect();
    info!("Downloading {} matrices...", urls.len());

    for url in urls.iter() {
        info!("Downloading: {url}");
        let stream = client.get(url).send().await?.bytes().await?;
        let mut gz = GzDecoder::new(&*stream);
        let mut out = Vec::new();
        gz.read_to_end(&mut out)?;
        let mut archive = tar::Archive::new(&out[..]);
        archive.unpack("data/suitesparse")?;
    }
    Ok(())
}
